## SiDG-ATRID
### Simulator for Data Generation - for Automatic Target Recognition, Identification and Detection

A simulation platform that enables the collection of high-fidelity imagery data, powered by Unreal Engine 5. The simulator supports multi-agent simulations using the AirSim API library for UAV controls and simulates commercial aircraft traffic. This framework allows for customized camera placements to record videos or photos and manage environmental conditions such as weather and lighting. Additionally, by leveraging the Cesium API for geospatial mapping, it can accurately recreate real-world environments, enhancing the realism and applicability of simulations. This integrated approach enhances the efficiency and effectiveness of synthetic data generation for detection tasks, enabling developers to easily configure simulations and collect diverse data.

[![SiDG-ATRID Demo Reel](https://img.youtube.com/vi/gmdaVOeZyiQ/0.jpg)](https://www.youtube.com/watch?v=gmdaVOeZyiQ)

## Config Folder

The `config` folder contains configuration files used to control various aspects of the simulation environment. Below are the descriptions of the JSON files and the included parameters found in this folder:

### sim_config.json

The `sim_config.json` file includes simulation control parameters. Currently, users can set the georeference origin, which serves as the center of the level. The available parameters are:

- **origin latitude**: Specifies the latitude of the origin point.
- **origin longitude**: Specifies the longitude of the origin point.
- **scale**: Determines the scale of the simulation.

### env_config.json

The `env_config.json` file includes parameters for configuring the simulation environment. The available parameters are:

- **TimeOfDay**: A value ranging from 0 to 2400, representing the time of day.
- **CloudCoverage**: A value ranging from 0.0 to 3.0, representing the amount of cloud coverage.
- **CloudSpeed**: A value ranging from 0.0 to 2.0, representing the speed of cloud movement.
- **Fog**: A value ranging from 0 to 10, representing the density of the fog.
- **CloudShadowEnable**: A boolean value (`true` or `false`) to enable or disable cloud shadows.
- **WeatherPreset**: A string value representing the weather preset. Possible values are:
  - `"Clear_Skies"`
  - `"Rain"`
  - `"Snow"`
- **WindDirection**: A value ranging from 0 to 400, representing the direction of the wind.
- **NightBrightness**: A value ranging from 0.0 to 2.0, representing the brightness during the night.
- **Dust**: A value ranging from 0.0 to 1.0, representing the amount of dust in the environment.
- **Stars Intensity**: A value ranging from 0.0 to 8.0, representing the intensity of the stars.

### Database

|  | Daytime & Clear | Night & Clear | Fog | Cloudy |
|----------|----------|----------|----------|----------|
| Multi-agent (Hexacopter, Fixed-wing UAV & Commercial Airliner)   | Row 1    | Row 1    | Row 1    | Row 1    |
| Hexacopter    | Row 2    | Row 2    | Row 2    | Row 2    |
| Fixed-wing UAV    | Row 3    | Row 3    | Row 3    | Row 3    |
| Ground Vehicle    | Row 4    | Row 4    | Row 4    | Row 4    |
| Commercial Airliner    | Row 5    | Row 5    | Row 5    | Row 5    |
