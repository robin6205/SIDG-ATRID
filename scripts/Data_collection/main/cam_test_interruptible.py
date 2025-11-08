import threading
import time
import sys
import os
import json
import re
import signal
from unrealcv import Client
import airsim
import math

BASE_SAVE_DIR = r"F:\SIDG-ATRID-Dataset\Train_set\test"

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config", "bo_config-city.json")

# Allowed characters for directory and file names
_SANITIZE_PATTERN = re.compile(r"[^A-Za-z0-9_\-]+")

def sanitize_label(label: str) -> str:
    """Return a filesystem-friendly label for directories and filenames."""
    return _SANITIZE_PATTERN.sub("_", label).strip("_") or "camera"

# Global flag to stop operations
stop_flag = threading.Event()
airsim_client = None
unrealcv_client = None

def signal_handler(signum, frame):
    """Handle SIGINT (Ctrl+C) and SIGTERM signals"""
    print("\n[INFO] Interrupt signal received! Stopping...")
    stop_flag.set()
    # Give a moment for normal cleanup, then force exit if still hanging
    def force_exit_after_delay():
        time.sleep(3)  # Wait 3 seconds for normal cleanup
        # If we're still here, cleanup didn't complete - force exit
        print("[WARNING] Cleanup taking too long, forcing exit...")
        cleanup_and_exit()
    
    # Start force exit timer in background
    force_thread = threading.Thread(target=force_exit_after_delay)
    force_thread.daemon = True
    force_thread.start()

def cleanup_and_exit():
    """Force cleanup and exit"""
    global airsim_client, unrealcv_client
    try:
        if airsim_client:
            try:
                print("[INFO] Forcing cleanup of AirSim client...")
                airsim_client.armDisarm(False)
                airsim_client.enableApiControl(False)
            except:
                pass
        if unrealcv_client:
            try:
                print("[INFO] Closing UnrealCV connection...")
                unrealcv_client.disconnect()
            except:
                pass
    except:
        pass
    print("[INFO] Force exiting...")
    os._exit(0)

def safe_request(client, command, timeout=10):
    """Make a request with timeout and interrupt checking"""
    if stop_flag.is_set():
        return None
    
    result = [None]
    exception = [None]
    
    def make_request():
        try:
            result[0] = client.request(command)
        except Exception as e:
            exception[0] = e
    
    # Start the request in a separate thread
    request_thread = threading.Thread(target=make_request)
    request_thread.daemon = True
    request_thread.start()
    
    # Wait for completion or timeout/interrupt
    start_time = time.time()
    while request_thread.is_alive():
        if stop_flag.is_set():
            print(f"[INFO] Request '{command}' interrupted by user")
            return None
        if time.time() - start_time > timeout:
            print(f"[ERROR] Request '{command}' timed out after {timeout} seconds")
            return None
        time.sleep(0.1)
    
    if exception[0]:
        raise exception[0]
    return result[0]

def wait_for_async_with_timeout(async_op, timeout=30, description="operation"):
    """Wait for an AirSim async operation with timeout and interrupt checking.
    
    AirSim async operations return msgpackrpc.future.Future objects.
    We need to call .join() to wait, but we wrap it in a thread to make it interruptible.
    """
    if stop_flag.is_set():
        return False
    
    result = [None]  # [True] = success, [False] = timeout/interrupt, [Exception] = error
    exception = [None]
    
    def wait_for_completion():
        """Run the blocking join() in a separate thread"""
        try:
            async_op.join()  # This blocks until operation completes
            if not stop_flag.is_set():
                result[0] = True
            else:
                result[0] = False
        except Exception as e:
            exception[0] = e
            result[0] = False
    
    # Start waiting in a separate thread
    wait_thread = threading.Thread(target=wait_for_completion)
    wait_thread.daemon = True
    wait_thread.start()
    
    # Poll for completion, timeout, or interrupt
    start_time = time.time()
    while wait_thread.is_alive():
        if stop_flag.is_set():
            print(f"[INFO] {description} interrupted by user")
            return False
        if time.time() - start_time > timeout:
            print(f"[WARNING] {description} timed out after {timeout} seconds")
            return False
        time.sleep(0.1)
    
    # Check for exceptions
    if exception[0]:
        print(f"[ERROR] {description} failed with error: {exception[0]}")
        return False
    
    return result[0] if result[0] is not None else False

def load_configuration(config_path: str = CONFIG_PATH) -> dict:
    """Load and return the JSON configuration for the data collection run."""
    with open(config_path, "r") as config_file:
        return json.load(config_file)


def parse_camera_ids(response: str) -> set:
    """Extract integer camera IDs from a response string."""
    ids = set()
    if not response:
        return ids
    for token in re.split(r"\s+", str(response).strip()):
        if not token:
            continue
        digits = re.findall(r"\d+", token)
        for value in digits:
            try:
                ids.add(int(value))
            except ValueError:
                continue
    return ids


def get_camera_ids(client: Client) -> set:
    """Query UnrealCV for the list of currently available camera IDs."""
    resp = safe_request(client, "vget /cameras", timeout=5)
    return parse_camera_ids(resp) if resp is not None else set()


def spawn_camera(client: Client, camera_name: str):
    """Spawn a new camera and return its assigned UnrealCV camera ID."""
    resp = safe_request(client, "vset /cameras/spawn", timeout=5)
    if resp is None:
        print(f"[ERROR] Failed to spawn camera '{camera_name}' - no response.")
        return None

    resp_str = str(resp).strip()
    print(f"[DEBUG] Spawn response for '{camera_name}': {resp_str}")
    
    # Try to extract camera ID from response
    # Response format is typically "FusionCameraActor_N" where N is the camera ID
    camera_id = None
    
    # Method 1: Check if response is a direct integer
    if resp_str.isdigit():
        camera_id = int(resp_str)
        print(f"[INFO] Camera '{camera_name}' spawned with ID {camera_id} (from direct response)")
        return camera_id
    
    # Method 2: Extract number from actor name like "FusionCameraActor_0"
    # Look for pattern like "_N" at the end of the string
    match = re.search(r'_(\d+)$', resp_str)
    if match:
        camera_id = int(match.group(1))
        print(f"[INFO] Camera '{camera_name}' spawned with ID {camera_id} (extracted from '{resp_str}')")
        return camera_id
    
    # Method 3: Try to find any number in the response
    digits = re.findall(r'\d+', resp_str)
    if digits:
        camera_id = int(digits[-1])  # Use the last number found
        print(f"[INFO] Camera '{camera_name}' spawned with ID {camera_id} (found number in response)")
        return camera_id

    print(f"[ERROR] Could not extract camera ID from response: {resp_str}")
    return None


def setup_static_camera(client: Client, camera_id: int, camera_config: dict) -> None:
    """Configure location and rotation for a static camera."""
    location = camera_config.get("location")
    if location:
        loc_cmd = (
            f"vset /camera/{camera_id}/location "
            f"{float(location.get('x', 0.0)):.6f} "
            f"{float(location.get('y', 0.0)):.6f} "
            f"{float(location.get('z', 0.0)):.6f}"
        )
        loc_resp = safe_request(client, loc_cmd, timeout=5)
        verify_loc = safe_request(client, f"vget /camera/{camera_id}/location", timeout=5)
        print(f"[INFO] Set location for camera {camera_id}: {loc_resp} (verify: {verify_loc})")
    else:
        print(f"[WARNING] Static camera {camera_id} missing location data.")

    rotation = camera_config.get("rotation")
    if rotation:
        rot_cmd = (
            f"vset /camera/{camera_id}/rotation "
            f"{float(rotation.get('pitch', 0.0)):.6f} "
            f"{float(rotation.get('yaw', 0.0)):.6f} "
            f"{float(rotation.get('roll', 0.0)):.6f}"
        )
        rot_resp = safe_request(client, rot_cmd, timeout=5)
        verify_rot = safe_request(client, f"vget /camera/{camera_id}/rotation", timeout=5)
        print(f"[INFO] Set rotation for camera {camera_id}: {rot_resp} (verify: {verify_rot})")
    else:
        print(f"[WARNING] Static camera {camera_id} missing rotation data.")


def attach_camera_to_drone(client: Client, camera_id: int, camera_config: dict, blueprint_name: str) -> None:
    """Attach a spawned camera to the drone using the provided offset and rotation.
    
    Command format: vrun ce cameraattach <camera_id> <blueprint_name> <offset_x> <offset_y> <offset_z> <pitch> <yaw> <roll>
    Example: vrun ce cameraattach 1 BP_DJIS900 200 10 50 0 -10 0
    """
    offset = camera_config.get("location", {})
    rotation = camera_config.get("rotation", {})

    offset_x = float(offset.get("x", 0.0))
    offset_y = float(offset.get("y", 0.0))
    offset_z = float(offset.get("z", 0.0))

    # compute yaw that actually faces the drone from this offset
    # For Unreal's axes (+X forward, +Y right, +Z up), the yaw that "looks back" 
    # from a camera at offset (x, y, z) toward the drone at the origin is:
    # Yaw (deg) = atan2(-y, -x) (convert to degrees, wrap to 0–360)
    yaw_unreal = math.degrees(math.atan2(-offset_y, -offset_x))
    yaw = (yaw_unreal + 360.0) % 360.0

    # compute pitch that actually faces the drone from this offset
    # Pitch (deg) = -atan2(z, sqrt(x^2 + y^2)) (negative because we look toward origin)
    # For camera below (z < 0): pitch should be positive (look up)
    # For camera above (z > 0): pitch should be negative (look down)
    horizontal_dist = math.hypot(offset_x, offset_y)
    pitch_unreal = -math.degrees(math.atan2(offset_z, horizontal_dist))
    pitch = pitch_unreal

    # keep the provided roll from config
    roll = float(rotation.get("roll", 0.0))

    attach_cmd = (
        f"vrun ce cameraattach {camera_id} {blueprint_name} "
        f"{offset_x:.6f} {offset_y:.6f} {offset_z:.6f}"
        f" {yaw:.6f} {pitch:.6f} {roll:.6f}"
    )
    
    print(f"[INFO] Executing attach command: {attach_cmd}")
    resp = safe_request(client, attach_cmd, timeout=5)
    print(f"[INFO] Attach command response for camera {camera_id}: {resp}")
    
    # Verify the attachment
    time.sleep(0.3)  # Small delay for attachment to take effect
    verify_loc = safe_request(client, f"vget /camera/{camera_id}/location", timeout=5)
    verify_rot = safe_request(client, f"vget /camera/{camera_id}/rotation", timeout=5)
    print(f"[INFO] Post-attach location for camera {camera_id}: {verify_loc}")
    print(f"[INFO] Post-attach rotation for camera {camera_id}: {verify_rot}")


def configure_cameras(unrealcv_client: Client, config: dict, base_save_dir: str):
    """Spawn and configure cameras based on the configuration file."""
    camera_entries = []
    camera_config = config.get("camera_config", {})
    drone_type = config.get("drone_config", {}).get("drone_type", "DJIS900")
    blueprint_name = f"BP_{drone_type}"

    # Spawn a dummy camera to consume any pre-existing camera 0 from AirSim
    print("[INFO] Spawning dummy camera to handle pre-existing camera IDs...")
    dummy_resp = safe_request(unrealcv_client, "vset /cameras/spawn", timeout=5)
    if dummy_resp:
        print(f"[INFO] Dummy camera spawned: {dummy_resp}")
        # Extract and print the ID but don't use it
        dummy_match = re.search(r'_(\d+)$', str(dummy_resp).strip())
        if dummy_match:
            print(f"[INFO] Dummy camera consumed ID: {dummy_match.group(1)}")
    time.sleep(0.5)  # Small delay for stability

    for camera_name, camera_data in camera_config.items():
        if stop_flag.is_set():
            break
            
        camera_type = camera_data.get("type", "").lower()
        safe_name = sanitize_label(camera_name)
        camera_dir = base_save_dir  # Save directly in the base directory
        os.makedirs(camera_dir, exist_ok=True)

        print(f"[INFO] Configuring camera '{camera_name}' (type: {camera_type})")
        
        camera_id = spawn_camera(unrealcv_client, camera_name)
        if camera_id is None:
            print(f"[WARNING] Failed to spawn camera '{camera_name}', skipping...")
            continue

        if camera_type == "staticcam" or camera_type == "staticcamera":
            print(f"[INFO] Setting up static camera '{camera_name}' (ID: {camera_id})")
            setup_static_camera(unrealcv_client, camera_id, camera_data)
        elif camera_type == "attachdrone":
            print(f"[INFO] Attaching camera '{camera_name}' (ID: {camera_id}) to drone")
            attach_camera_to_drone(unrealcv_client, camera_id, camera_data, blueprint_name)
        else:
            print(f"[WARNING] Unknown camera type '{camera_type}' for '{camera_name}'. Skipping configuration.")
            continue

        camera_entries.append({
            "name": camera_name,
            "id": camera_id,
            "type": camera_type,
            "save_dir": camera_dir,
            "frame_count": 0,
            "label": safe_name
        })

    # Summary of configured cameras
    print(f"\n[INFO] ===== Camera Configuration Summary =====")
    print(f"[INFO] Total cameras configured: {len(camera_entries)}")
    for entry in camera_entries:
        print(f"[INFO]   - {entry['name']} (ID: {entry['id']}, Type: {entry['type']})")
    print(f"[INFO] ==========================================\n")
    
    # Small delay to let cameras stabilize
    if camera_entries:
        print("[INFO] Waiting 1 second for cameras to stabilize...")
        time.sleep(1)
    
    return camera_entries

def capture_image(client, cam_id, save_dir, frame_num, camera_label):
    """Capture a single image from the specified camera."""
    if stop_flag.is_set():
        return None

    filename = os.path.join(save_dir, f"{camera_label}_frame_{frame_num:04d}.png")
    cmd = f"vget /camera/{cam_id}/lit {filename}"

    resp = safe_request(client, cmd, timeout=5)
    if resp and "error" not in str(resp).lower():
        print(f"[SUCCESS] Frame {frame_num} captured from camera {cam_id} ({camera_label})")
        return filename
    else:
        print(f"[WARNING] Failed to capture frame {frame_num} from camera {cam_id} ({camera_label}): {resp}")
        return None

def main():
    global airsim_client, unrealcv_client
    
    # Register signal handlers for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)

    print("[INFO] Starting AirSim capture run (Press Ctrl+C to stop)")

    # Base directory to save images (fixed as per requirements)
    base_save_dir = BASE_SAVE_DIR
    os.makedirs(base_save_dir, exist_ok=True)

    camera_entries = []

    try:
        # Connect to AirSim
        print("[INFO] Connecting to AirSim...")
        airsim_client = airsim.MultirotorClient()
        airsim_client.confirmConnection()
        airsim_client.enableApiControl(True)
        airsim_client.armDisarm(True)
        print("[SUCCESS] Connected to AirSim")

        if stop_flag.is_set():
            return

        # Connect to UnrealCV
        print("[INFO] Connecting to UnrealCV server...")
        unrealcv_client = Client(("127.0.0.1", 9000))
        unrealcv_client.connect()

        if stop_flag.is_set():
            return

        if not unrealcv_client.isconnected():
            print("[ERROR] UnrealCV server not connected. Make sure Unreal is running.")
            return

        print("[SUCCESS] Connected to UnrealCV server")

        # Load configuration
        try:
            config = load_configuration(CONFIG_PATH)
            print(f"[INFO] Loaded configuration from {CONFIG_PATH}")
        except FileNotFoundError:
            print(f"[WARNING] Configuration file not found at {CONFIG_PATH}. Proceeding with default settings.")
            config = {}
        except json.JSONDecodeError as exc:
            print(f"[ERROR] Failed to parse configuration file: {exc}")
            return

        data_collection_cfg = config.get("data_collection", {})
        # Use the hardcoded BASE_SAVE_DIR instead of config override
        print(f"[INFO] Using base output directory: {base_save_dir}")

        # Spawn and configure cameras based on config
        camera_entries = configure_cameras(unrealcv_client, config, base_save_dir)
        if not camera_entries:
            print("[WARNING] No cameras configured; falling back to AirSim default camera 0.")
            fallback_dir = base_save_dir
            camera_entries.append({
                "name": "camera0",
                "id": 0,
                "type": "default",
                "save_dir": fallback_dir,
                "frame_count": 0,
                "label": sanitize_label("camera0")
            })

        # Flight parameters
        flight_duration = 10.0  # seconds
        frame_rate_value = data_collection_cfg.get("frame_rate")
        if frame_rate_value:
            try:
                capture_interval = max(0.01, 1.0 / float(frame_rate_value))
            except (TypeError, ValueError):
                print(f"[WARNING] Invalid frame_rate '{frame_rate_value}' in config; using default interval.")
                capture_interval = 0.5
        else:
            capture_interval = 0.5  # seconds between capture cycles
        takeoff_altitude = -30.0  # negative is up in AirSim (30m)
        forward_velocity = 5.0  # m/s

        # Takeoff
        print(f"[INFO] Taking off to altitude {abs(takeoff_altitude)} m...")
        takeoff_op = airsim_client.takeoffAsync()
        if not wait_for_async_with_timeout(takeoff_op, timeout=30, description="Takeoff"):
            return

        if stop_flag.is_set():
            return

        # Move to target altitude quickly
        print("[INFO] Moving to flight altitude...")
        move_op = airsim_client.moveToZAsync(takeoff_altitude, 6)
        if not wait_for_async_with_timeout(move_op, timeout=30, description="Move to altitude"):
            return

        if stop_flag.is_set():
            return

        print(f"[INFO] Beginning forward flight for {flight_duration} seconds")
        print(f"[INFO] Capturing images every {capture_interval} seconds across {len(camera_entries)} cameras")

        # Begin forward motion
        airsim_client.moveByVelocityAsync(forward_velocity, 0, 0, flight_duration)

        # Capture images during flight
        start_time = time.time()
        next_capture_time = start_time
        is_paused = False
        
        try:
            while time.time() - start_time < flight_duration:
                if stop_flag.is_set():
                    break

                # Check if it's time to capture
                current_time = time.time()
                if current_time >= next_capture_time:
                    # Pause simulation before capturing
                    print(f"[INFO] Pausing simulation for image capture...")
                    airsim_client.simPause(True)
                    is_paused = True
                    
                    # Small delay to ensure simulation is paused
                    time.sleep(0.1)
                    
                    # Capture images from all cameras while paused
                    print(f"[INFO] Capturing images from {len(camera_entries)} cameras...")
                    for entry in camera_entries:
                        if stop_flag.is_set():
                            break
                        capture_image(
                            unrealcv_client,
                            entry["id"],
                            entry["save_dir"],
                            entry["frame_count"],
                            entry["label"]
                        )
                        entry["frame_count"] += 1
                    
                    # Unpause simulation after all captures are complete
                    print(f"[INFO] Resuming simulation...")
                    airsim_client.simPause(False)
                    is_paused = False
                    
                    # Update next capture time
                    next_capture_time = current_time + capture_interval
                else:
                    # Sleep a short time to avoid busy waiting
                    time.sleep(0.01)
        finally:
            # Ensure simulation is unpaused if we exit the loop while paused
            if is_paused and airsim_client:
                try:
                    print("[INFO] Unpausing simulation before exiting capture loop...")
                    airsim_client.simPause(False)
                except:
                    pass

        if not stop_flag.is_set():
            total_frames = sum(entry["frame_count"] for entry in camera_entries)
            print(f"\n[SUCCESS] ===== Capture Complete =====")
            print(f"[SUCCESS] Total frames captured: {total_frames} across {len(camera_entries)} cameras")
            print(f"[INFO] Save directory: {base_save_dir}")
            print(f"[INFO] Frame breakdown by camera:")
            for entry in camera_entries:
                print(
                    f"[INFO]   - {entry['name']} (ID {entry['id']}, Type: {entry['type']}): "
                    f"{entry['frame_count']} frames"
                )
            print(f"[SUCCESS] ============================\n")
        else:
            print(f"\n[INFO] Capture interrupted by user")
            total_frames = sum(entry["frame_count"] for entry in camera_entries)
            print(f"[INFO] Partial frames captured: {total_frames}")
            for entry in camera_entries:
                if entry['frame_count'] > 0:
                    print(f"[INFO]   - {entry['name']}: {entry['frame_count']} frames")

        # Hover in place
        print("[INFO] Hovering in place...")
        hover_op = airsim_client.hoverAsync()
        wait_for_async_with_timeout(hover_op, timeout=10, description="Hover")

        if stop_flag.is_set():
            return

        # Land
        print("[INFO] Landing...")
        land_op = airsim_client.landAsync()
        wait_for_async_with_timeout(land_op, timeout=30, description="Land")

        print("[SUCCESS] All operations completed!")

    except KeyboardInterrupt:
        print("\n[INFO] Main thread interrupted")
        stop_flag.set()
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        stop_flag.set()
    finally:
        # Cleanup
        print("[INFO] Starting cleanup...")
        if airsim_client:
            try:
                # Unpause simulation if it's paused
                try:
                    if airsim_client.simIsPause():
                        print("[INFO] Unpausing simulation...")
                        airsim_client.simPause(False)
                except:
                    pass
                
                print("[INFO] Disarming and disabling API control...")
                airsim_client.armDisarm(False)
                airsim_client.enableApiControl(False)
            except Exception as exc:
                print(f"[WARNING] AirSim cleanup error: {exc}")
        
        if unrealcv_client:
            try:
                print("[INFO] Closing UnrealCV connection...")
                unrealcv_client.disconnect()
            except Exception as exc:
                print(f"[WARNING] UnrealCV cleanup error: {exc}")

        stop_flag.set()
        print("[INFO] Cleanup completed")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Script interrupted by user")
        stop_flag.set()
        cleanup_and_exit()
    except SystemExit:
        # Allow os._exit to work
        raise
    except Exception as e:
        print(f"[ERROR] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        cleanup_and_exit()
    finally:
        print("[INFO] Script finished")
        sys.exit(0)
