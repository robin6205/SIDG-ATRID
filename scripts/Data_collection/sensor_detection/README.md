# Sensor Detection Task

This directory contains scripts for generating and testing sensor configurations and placements for the drone detection sensor placement problem. The workflow simulates various camera configurations in different weather conditions to evaluate optimal sensor placement strategies.

## Overview

The sensor detection task generates comprehensive test scenarios by:
1. Creating camera configurations with varying quality levels and weather conditions
2. Generating drone waypoints for urban environments
3. Running simulations to collect data for sensor placement optimization

## Workflow

The complete workflow consists of three main scripts that should be executed in sequence:

### 1. Generate Sensor Configurations → 2. Generate Waypoints → 3. Run Simulation

```
generate_scenarios_fixed_cams.py → waypoint_gen.py → full_data_collection_visible_area.py
```

## Scripts Description

### 1. `generate_scenarios_fixed_cams.py`

**Purpose**: Generates weather configurations and camera configurations with placements for testing sensor detection scenarios.

**What it does**:
- Creates 4 camera configurations with quality gradient (1=best, 4=worst)
- Generates 100 unique scenarios per environment condition
- Creates 4 different weather environments (clear, moderate, poor, night)
- Outputs JSON configuration files for each scenario

**Camera Quality Levels**:
- **Camera 1**: Best quality (4K resolution, FOV 30°, exposure +3.0, focal 10.0)
- **Camera 2**: Good quality (1080p resolution, FOV 60°, exposure +1.0, focal 1000.0)
- **Camera 3**: Poor quality (720p resolution, FOV 90°, exposure -1.0, focal 5000.0)
- **Camera 4**: Worst quality (480p resolution, FOV 120°, exposure -3.0, focal 2000.0)

**Example execution**:
```bash
# Generate camera scenarios and configurations
python generate_scenarios_fixed_cams.py
```

**Output**: Creates configuration files in `scripts/Data_collection/data_collection_config/camera_config/`

---

### 2. `waypoint_gen.py`

**Purpose**: Generates drone waypoints for the urban environment and visualizes sensor field of view coverage.

**What it does**:
- Creates a grid of waypoints covering urban streets (vertical and horizontal)
- Generates intersection waypoints for comprehensive coverage
- Visualizes camera positions and field of view cones
- Exports waypoint data in mission-ready JSON format
- Optionally highlights detected waypoints from previous runs

**Example executions**:
```bash
# Basic waypoint generation
python waypoint_gen.py

# Generate waypoints with detection data visualization
python waypoint_gen.py --detection

# Show waypoint numbers on the visualization
python waypoint_gen.py --count

# Generate with both detection highlighting and waypoint numbers
python waypoint_gen.py --detection --count
```

**Parameters**:
- `--detection`: Enable loading detection data and highlighting detected waypoints
- `--count`: Display waypoint numbers on each point in the visualization

**Output**: Creates `generated_mission_data_debug_2.json` with waypoint configurations

---

### 3. `full_data_collection_visible_area.py`

**Purpose**: Runs the simulation to collect data using the generated camera configurations and waypoints.

**What it does**:
- Connects to UnrealCV and AirSim for simulation control
- Sets up cameras with specified configurations (position, rotation, FOV, resolution)
- Controls drone movement through generated waypoints
- Captures RGB and object mask images at each waypoint
- Supports both sequential (per-camera) and parallel (all-cameras) collection modes
- Optionally saves drone/camera state data for analysis

**Example executions**:
```bash
# Basic data collection (sequential mode)
python full_data_collection_visible_area.py --config "path/to/config.json"

# Run with state saving enabled
python full_data_collection_visible_area.py --config "path/to/config.json" --state

# Run in parallel mode (all cameras capture at each waypoint)
python full_data_collection_visible_area.py --config "path/to/config.json" --parallel

# Run with visualization and state saving
python full_data_collection_visible_area.py --config "path/to/config.json" --state --visualize-line

# Use specific configuration file
python full_data_collection_visible_area.py --config "scripts/Data_collection/data_collection_config/config8-brushify-lake.json" --parallel --state
```

**Parameters**:
- `--config`: Path to the configuration JSON file (required)
- `--state`: Enable saving of drone and camera state data to JSON files
- `--visualize-line`: Draw arrows from drone to destination points (visualization)
- `--parallel`: Run in parallel mode where all cameras capture at each waypoint

**Collection Modes**:
- **Sequential Mode** (default): Complete full mission for each camera separately
- **Parallel Mode** (`--parallel`): All cameras capture images at each waypoint simultaneously

## Complete Workflow Example

Here's a complete example of running the entire sensor detection pipeline:

```bash
# Step 1: Generate camera scenarios and configurations
cd scripts/Data_collection/sensor_detection
python generate_scenarios_fixed_cams.py

# Step 2: Generate waypoints for the urban environment
python waypoint_gen.py --count

# Step 3: Run the simulation data collection
python full_data_collection_visible_area.py \
    --config "scripts/Data_collection/data_collection_config/config8-brushify-lake.json" \
    --parallel \
    --state
```

## Configuration Files

### Camera Configuration Format
Each camera scenario is stored as JSON with the following structure:
```json
[
  [camera_id, fov, width, height, exposure_level, focal_length, loc_1, loc_2, loc_3, loc_4, detection_output]
]
```

### Mission Configuration
Waypoint missions are stored with drone waypoints and camera specifications:
```json
{
  "drone_config": {...},
  "camera_config": {...},
  "drones": [...],
  "cameras": [...],
  "data_collection": {...}
}
```

## Output Data

The simulation generates:
- **RGB Images**: `camera_dir/rgb/frameindex_timestamp_cameraid_lit.png`
- **Object Masks**: `camera_dir/mask/frameindex_timestamp_cameraid_object_mask.png`
- **State Data** (optional): `camera_dir/state/frameindex_timestamp_cameraid_state.json`
- **Camera Configs**: `camera_dir/camera_config.json`
- **Weather Configs**: `camera_dir/formatted_ultra_dynamic_sky.json`

## Dependencies

- Python >= 3.8
- UnrealCV
- AirSim
- NumPy
- Matplotlib
- JSON

## Environment Requirements

- Unreal Engine 5.4+ with AirSim plugin
- UnrealCV server running on port 9000
- AirSim multirotor simulation environment

## Usage Notes

1. **Run scripts in order**: Configuration → Waypoints → Simulation
2. **Ensure simulation environment is running** before executing `full_data_collection_visible_area.py`
3. **Use parallel mode** for efficient data collection across multiple cameras
4. **Enable state saving** for detailed analysis of drone and camera parameters
5. **Check output directories** are accessible and have sufficient storage space

For detailed configuration options and troubleshooting, refer to individual script documentation and the generated configuration files in `camera_config/README.md`.
