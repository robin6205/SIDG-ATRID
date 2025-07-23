README: Scenario Configuration Structure

This dataset contains scenario configuration files for four environment conditions:

Folders:
- z1/: Clear bright sky
- z2/: Foggy, slightly overcast
- z3/: Rainy evening
- z4/: Night with fog and light rain

Each folder contains:
1. env.json
   - TimeOfDay: Integer from 0-2400 (representing hour of day)
   - fog: Integer from 0-100
   - rain: Integer from 0-50
   - cloud: Integer from 0-100 (cloud density)

2. subset_#.json (1-100)
   - Each file contains a list of camera configurations
   - Each configuration is a flat list of 11 elements:

     [camera_id, fov, resolution_width, resolution_height, exposure_level, focal_length, loc1, loc2, loc3, loc4, detection_output]

     where:
     - camera_id: 1-4
     - fov: Field of view (degrees)
     - resolution_width, resolution_height: Integer resolution
     - exposure_level: Float from -3.0 to 3.0 (camera exposure setting)
     - focal_length: Float focal length setting
     - loc1-loc4: One-hot location vector for one of four possible positions
     - detection_output: Empty list [] that will be populated later with detection results

   Example:
   [
     [1, 30, 3840, 2160, 3.0, 10.0, 1, 0, 0, 0, []],
     [2, 60, 1920, 1080, 1.0, 1000.0, 0, 1, 0, 0, []]
   ]
   means camera 1 is at location 1, camera 2 at location 2, both with empty detection outputs.

The same 100 camera subset files are reused across all four z folders.
