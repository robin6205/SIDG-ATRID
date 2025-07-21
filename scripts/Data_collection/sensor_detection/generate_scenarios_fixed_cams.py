import os
import json
import random
import itertools
"""
===============================================================================
File Name   : generate_scenarios_fixed_cams.py
Description : Generates unique camera scenarios for fixed camera configurations
              with different field of views, resolutions, exposure levels, and 
              focal lengths. Creates combinations of camera setups and environment 
              conditions for sensor detection experiments. Camera 1 has the best
              settings (highest quality) and camera 4 has the worst settings.
Author      : Josh Chang <chang529@purdue.edu>
Created On  : 2025-07-21
Last Updated: 2025-07-21
Version     : 1.0.0

Usage       : python generate_scenarios_fixed_cams.py
Example     : python generate_scenarios_fixed_cams.py

Dependencies:
    - Python >= 3.8
    - random, itertools, json, os

Notes:
    - Generates 100 unique scenarios from all possible combinations
    - Each scenario includes camera configurations and environment conditions
    - Camera quality gradient: 1 (best) to 4 (worst)
    - Exposure levels range from -3 to 3
    - Focal lengths range from 10 to 2000
    - Outputs JSON files with scenario configurations
===============================================================================
"""


# Define camera configurations with quality gradient (1=best, 4=worst)
# Format: [fov, (width, height), exposure_level, focal_length]
camera_configs = {
    1: [30, (3840, 2160), 3.0, 1000.0],      # Best: highest resolution, best exposure, best focal length
    2: [60, (1920, 1080), 1.0, 2000.0],    # Good: good resolution, good exposure, good focal length  
    3: [90, (1280, 720), -1.0, 3000.0],    # Poor: lower resolution, poor exposure, poor focal length
    4: [120, (640, 480), -3.0, 4000.0]     # Worst: lowest resolution, worst exposure, worst focal length
}

# Possible locations (one-hot, flattened)
location_vectors = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
]

# All unique scenarios: avoid (cam1@loc1, cam2@loc2) == (cam2@loc2, cam1@loc1)
unique_scenarios_set = set()
unique_scenarios = []

for r in range(2, 5):  # 2 to 4 cameras
    for cam_ids in itertools.combinations(camera_configs.keys(), r):
        for locs in itertools.permutations(location_vectors, r):
            scenario = []
            normalized = []
            for cam_id, loc in zip(cam_ids, locs):
                fov, (w, h), exposure, focal = camera_configs[cam_id]
                flat = [cam_id, fov, w, h, exposure, focal, *loc, []]  # Added empty list for detection output
                scenario.append(flat)
                normalized.append((cam_id, tuple(loc)))  # for duplicate check
            normalized_key = tuple(sorted(normalized))  # sorted by cam id + loc
            if normalized_key not in unique_scenarios_set:
                unique_scenarios_set.add(normalized_key)
                unique_scenarios.append(scenario)

# Shuffle and pick 100 unique scenarios
print(f"Total unique scenarios generated: {len(unique_scenarios)}")
random.shuffle(unique_scenarios)
selected_scenarios = unique_scenarios[:100]

# Define environments
env_conditions = {
    "z1": {"TimeOfDay": 1200, "fog": 0, "rain": 0, "cloud": 20},
    "z2": {"TimeOfDay": 1000, "fog": random.randint(30, 50), "rain": random.randint(0, 10), "cloud": random.randint(40, 60)},
    "z3": {"TimeOfDay": 1930, "fog": random.randint(10, 30), "rain": random.randint(10, 30), "cloud": random.randint(70, 90)},
    "z4": {"TimeOfDay": 2300, "fog": random.randint(70, 100), "rain": random.randint(30, 50), "cloud": random.randint(90, 100)}
}

# Output
base_dir = r"C:\Users\Josh\Desktop\PhD\Research\SiDG-ATRID\git\SIDG-ATRID\scripts\Data_collection\data_collection_config\camera_config"
os.makedirs(base_dir, exist_ok=True)

for z_key, env in env_conditions.items():
    folder = os.path.join(base_dir, z_key)
    os.makedirs(folder, exist_ok=True)

    with open(os.path.join(folder, "env.json"), "w") as f:
        json.dump(env, f, indent=2)

    for i, scenario in enumerate(selected_scenarios, 1):
        with open(os.path.join(folder, f"subset_{i}.json"), "w") as f:
            json.dump(scenario, f, indent=2)

print(f"Generated 100 unique scenarios per environment in '{base_dir}'")
