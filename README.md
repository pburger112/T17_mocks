
---

# LensingMocks

A Python class designed for generating mock lensing data, particularly for creating lensing maps, shear catalogs, and simulating galaxy positions based on cosmological simulations. This class provides tools to download necessary simulation data, process it, and generate catalogs suitable for weak lensing analyses.

## Table of Contents

- [Features](#features)
- [Dependencies](#dependencies)
- [Data Downloads](#data-downloads)
- [Contributing](#contributing)


## Features

- **Data Downloading**: Automatically downloads lensing and density files required for computations.
- **Map Loading**: Efficiently loads convergence (kappa) and shear (gamma1, gamma2) maps from binary files.
- **Rotation Utilities**: Functions to rotate galaxy coordinates and shear components.
- **Lensing Map Creation**: Generates lensing maps by combining data across redshift bins.
- **Shear Catalog Generation**: Creates shear catalogs with added observational noise and intrinsic ellipticity.
- **Galaxy Position Simulation**: Simulates galaxy positions by sampling from the density field.



## Dependencies

- Python 3.6 or higher
- `numpy`
- `healpy`
- `astropy`
- `scikit-learn`
- `tqdm`
- `wget`

## Data Downloads

The class interacts with data from the Takahashi et al. full-sky gravitational lensing simulation:

- Lensing maps and density shells are downloaded from the [Hirosaki University website](http://cosmo.phys.hirosaki-u.ac.jp/takahasi/allsky_raytracing/).
- Ensure you have sufficient disk space to store the downloaded data.

## Contributing

Contributions are welcome! If you find bugs or have suggestions for improvements, please open an issue or submit a pull request.

To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes with clear messages.
4. Push to your fork and submit a pull request.

