from unrealcv import Client
import os
import signal
import sys

def timeout_handler(signum, frame):
    print("\n[ERROR] Operation timed out! Forcing exit.")
    sys.exit(1)

# Set up timeout handler (works on Unix/Linux, limited on Windows)
try:
    signal.signal(signal.SIGALRM, timeout_handler)
except AttributeError:
    print("[WARNING] Timeout handling not available on Windows")

print("[INFO] Attempting to connect to UnrealCV server...")
try:
    client = Client(('127.0.0.1', 9000))
    print("[INFO] Client created, attempting connection...")
    
    # Try to set a timeout if possible
    try:
        signal.alarm(10)  # 10 second timeout
    except AttributeError:
        pass
    
    client.connect()
    
    try:
        signal.alarm(0)  # Cancel timeout
    except AttributeError:
        pass
    
    print("[INFO] Connection attempt completed")
    
    if not client.isconnected():
        print("[ERROR] UnrealCV server not connected. Make sure Unreal is running.")
        sys.exit(1)
    else:
        print("[SUCCESS] Connected to UnrealCV server")
        
except KeyboardInterrupt:
    print("\n[INFO] Connection interrupted by user")
    sys.exit(0)
except Exception as e:
    print(f"[ERROR] Connection failed: {e}")
    sys.exit(1)

# Directory to save images
save_dir = r"F:\SIDG-ATRID-Dataset\Train_set\test"
os.makedirs(save_dir, exist_ok=True)

# # ---- Spawn cameras 0, 1, and 2 ----
# resp0 = client.request('vset /cameras/spawn')
# print(f"Spawn camera 0: {resp0}")
# resp1 = client.request('vset /cameras/spawn')
# print(f"Spawn camera 1: {resp1}")
# resp2 = client.request('vset /cameras/spawn')
# print(f"Spawn camera 2: {resp2}")

# # ---- Camera 1 pose ----
# cam1_loc = (-30618.376694, -80232.128444, 346.959643)
# cam1_rot = (30.183478, 90.799996, 0.0)
# # ---- Camera 2 pose ----
# cam2_loc = (-41053.591280, -72741.444300, 305.686513)
# cam2_rot = (29.383478, -1.200005, 0.0)

# # ---- Set poses for cameras after spawning ----
# client.request(f"vset /camera/0/location {cam1_loc[0]} {cam1_loc[1]} {cam1_loc[2]}")
# client.request(f"vset /camera/0/rotation {cam1_rot[0]} {cam1_rot[1]} {cam1_rot[2]}")
# client.request(f"vset /camera/1/location {cam2_loc[0]} {cam2_loc[1]} {cam2_loc[2]}")
# client.request(f"vset /camera/1/rotation {cam2_rot[0]} {cam2_rot[1]} {cam2_rot[2]}")

# print("Spawned camera 0 and 1 at desired locations.")

# # ---- Call Blueprint attach for cam3 ----
# # (Make sure your Level Blueprint event name is lowercase "cameraattach")
resp = client.request("vrun ce cameraattach 0 BP_DJIS900 1300 10 200")
# print("Attach response:", resp)

# ---- Capture images ----
def capture_rgb(cam_id, filename):
    try:
        print(f"[INFO] Attempting to capture from camera {cam_id}...")
        # 'lit' = RGB render
        cmd = f"vget /camera/{cam_id}/lit {filename}"
        print(f"[INFO] Sending command: {cmd}")
        
        # Try to set a timeout if possible
        try:
            signal.alarm(15)  # 15 second timeout for capture
        except AttributeError:
            pass
        
        resp = client.request(cmd)
        
        try:
            signal.alarm(0)  # Cancel timeout
        except AttributeError:
            pass
        
        print(f"[SUCCESS] Captured from camera {cam_id}: {resp}")
        return resp
    except KeyboardInterrupt:
        print(f"\n[INFO] Capture from camera {cam_id} interrupted by user")
        raise
    except Exception as e:
        print(f"[ERROR] Failed to capture from camera {cam_id}: {e}")
        return f"error {e}"

try:
    # Example: Uncomment your camera creation and pose logic if/when needed
    # print("[INFO] Spawning cameras...")
    # resp0 = client.request('vset /cameras/spawn')
    # print(f"Spawn camera 0: {resp0}")
    # resp1 = client.request('vset /cameras/spawn')
    # print(f"Spawn camera 1: {resp1}")
    # resp2 = client.request('vset /cameras/spawn')
    # print(f"Spawn camera 2: {resp2}")
    
    # ---- Uncomment and use pose setting/capture logic as needed ----
    # print("[INFO] Setting camera poses...")
    # client.request("vset /camera/0/location ...")
    # ...
    
    print("[INFO] Starting image capture...")
    capture_rgb(1, os.path.join(save_dir, "cam1_rgb.png"))
    # capture_rgb(1, os.path.join(save_dir, "cam2_rgb.png"))
    # capture_rgb(2, os.path.join(save_dir, "cam3_rgb.png"))
    print("All captures done.")
except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user. Exiting now.")


