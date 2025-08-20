# NetCDF Boundary Conditions in Geodynamo.jl

This document describes how to use NetCDF files to specify temperature and compositional boundary conditions in Geodynamo.jl simulations.

## Overview

Geodynamo.jl supports reading boundary conditions from NetCDF files, providing:

- **Flexible boundary specification**: Support for both inner (CMB) and outer (surface) boundaries
- **Time-dependent boundaries**: Evolving boundary conditions during simulation
- **Automatic interpolation**: Handles grid mismatches between NetCDF files and simulation grid
- **Comprehensive validation**: Ensures compatibility and data integrity
- **Easy integration**: Seamless incorporation into existing simulation workflows

## Quick Start

```julia
using Geodynamo

# Load temperature boundary conditions from separate files
temp_boundaries = load_temperature_boundaries("cmb_temp.nc", "surface_temp.nc")

# Load compositional boundary conditions
comp_boundaries = load_composition_boundaries("cmb_composition.nc", "surface_composition.nc")

# Apply to simulation fields
apply_netcdf_temperature_boundaries!(temp_field, temp_boundaries)
apply_netcdf_composition_boundaries!(comp_field, comp_boundaries)
```

## NetCDF File Format Requirements

### File Structure

Each boundary NetCDF file should contain:

#### Required Variables
- **Data variable**: The main field data (temperature, composition, etc.)
- **Coordinate variables** (optional but recommended):
  - `theta` or `lat`/`latitude`: Colatitude [0,π] or latitude [-90,90] in degrees
  - `phi` or `lon`/`longitude`: Longitude [0,2π] or [0,360] in degrees/radians
  - `time` (optional): Time coordinate for time-dependent boundaries

#### Required Dimensions
- **Spatial dimensions**: `lat` × `lon` (2D) or `lat` × `lon` × `time` (3D)
- **Coordinate dimensions**: Match the spatial dimensions

### Example NetCDF Structure

```
dimensions:
    lat = 64 ;
    lon = 128 ;
    time = 10 ;  // Optional for time-dependent data

variables:
    double theta(lat) ;
        theta:long_name = "Colatitude" ;
        theta:units = "radians" ;
        
    double phi(lon) ;
        phi:long_name = "Longitude" ;
        phi:units = "radians" ;
        
    double time(time) ;  // Optional
        time:long_name = "Time" ;
        time:units = "dimensionless_time" ;
        
    double temperature(lat, lon) ;  // or (lat, lon, time)
        temperature:long_name = "Temperature boundary condition" ;
        temperature:units = "K" ;
        temperature:description = "CMB temperature boundary" ;
```

## Creating NetCDF Boundary Files

### Using the Sample Generator

Geodynamo.jl includes a sample NetCDF generator:

```bash
julia examples/create_sample_netcdf_boundaries.jl
```

This creates example files:
- `cmb_temp.nc` / `surface_temp.nc`: Time-independent temperature boundaries
- `cmb_temp_timedep.nc` / `surface_temp_timedep.nc`: Time-dependent temperature boundaries  
- `cmb_composition.nc` / `surface_composition.nc`: Compositional boundaries

### Manual Creation

#### Time-Independent Temperature Boundary

```julia
using NCDatasets

# Create CMB temperature boundary
NCDataset("cmb_temp.nc", "c") do ds
    # Define dimensions
    defDim(ds, "lat", 64)
    defDim(ds, "lon", 128)
    
    # Define coordinates
    defVar(ds, "theta", Float64, ("lat",))
    defVar(ds, "phi", Float64, ("lon",))
    
    # Define temperature variable
    temp_var = defVar(ds, "temperature", Float64, ("lat", "lon"), 
                      attrib=Dict("units" => "K", 
                                 "long_name" => "CMB temperature"))
    
    # Write coordinate data
    theta = range(0, π, length=64)
    phi = range(0, 2π, length=129)[1:128]  # Exclude endpoint
    
    ds["theta"][:] = theta
    ds["phi"][:] = phi
    
    # Create temperature pattern
    temperature = zeros(64, 128)
    T_base = 4000.0  # Base CMB temperature [K]
    
    for (i, th) in enumerate(theta)
        for (j, ph) in enumerate(phi)
            # Add spherical harmonic perturbation
            Y11 = sin(th) * cos(ph)  # Y₁₁ mode
            temperature[i, j] = T_base + 200.0 * Y11
        end
    end
    
    # Write temperature data
    temp_var[:] = temperature
    
    # Add global attributes
    ds.attrib["title"] = "CMB Temperature Boundary"
    ds.attrib["description"] = "Core-mantle boundary temperature"
end
```

#### Time-Dependent Boundary

```julia
NCDataset("cmb_temp_timedep.nc", "c") do ds
    defDim(ds, "lat", 64)
    defDim(ds, "lon", 128) 
    defDim(ds, "time", 100)  # 100 time steps
    
    # Define variables with time dimension
    defVar(ds, "theta", Float64, ("lat",))
    defVar(ds, "phi", Float64, ("lon",))
    defVar(ds, "time", Float64, ("time",))
    
    temp_var = defVar(ds, "temperature", Float64, ("lat", "lon", "time"))
    
    # Write coordinates
    ds["theta"][:] = range(0, π, length=64)
    ds["phi"][:] = range(0, 2π, length=129)[1:128]
    ds["time"][:] = range(0, 10, length=100)  # Time span
    
    # Write time-dependent temperature
    temperature = zeros(64, 128, 100)
    
    for (k, t) in enumerate(ds["time"][:])
        for (i, th) in enumerate(ds["theta"][:])
            for (j, ph) in enumerate(ds["phi"][:])
                # Rotating thermal pattern
                phase = 2π * t / 10  # Complete rotation every 10 time units
                Y11 = sin(th) * cos(ph + phase)
                temperature[i, j, k] = 4000.0 + 200.0 * Y11
            end
        end
    end
    
    temp_var[:] = temperature
end
```

## Loading Boundary Conditions

### Basic Loading

```julia
using Geodynamo

# Load temperature boundaries (inner and outer files)
temp_boundaries = load_temperature_boundaries("cmb_temp.nc", "surface_temp.nc")

# Load composition boundaries
comp_boundaries = load_composition_boundaries("cmb_comp.nc", "surface_comp.nc")

# Specify precision (default: Float64)
temp_boundaries = load_temperature_boundaries("cmb_temp.nc", "surface_temp.nc", 
                                             precision=Float32)
```

### Inspecting Loaded Boundaries

```julia
# Print detailed information
print_boundary_info(temp_boundaries)

# Get statistical summary
inner_stats = get_boundary_statistics(temp_boundaries.inner_boundary)
outer_stats = get_boundary_statistics(temp_boundaries.outer_boundary)

println("Inner boundary temperature range: [$(inner_stats["min"]), $(inner_stats["max"])] $(inner_stats["units"])")
println("Outer boundary temperature mean: $(outer_stats["mean"]) $(outer_stats["units"])")
```

### Validation

```julia
# Create SHTns configuration
config = create_optimized_config(32, 32, nlat=64, nlon=128)

# Validate compatibility
validate_netcdf_temperature_compatibility(temp_boundaries, config)
validate_netcdf_composition_compatibility(comp_boundaries, config)
```

## Applying Boundary Conditions

### Basic Application

```julia
# Create simulation fields
domain = create_radial_domain(0.35, 1.0, 64)
temp_field = create_shtns_temperature_field(Float64, config, domain)
comp_field = create_shtns_composition_field(Float64, config, domain)

# Apply boundary conditions at simulation start
apply_netcdf_temperature_boundaries!(temp_field, temp_boundaries, 0.0)
apply_netcdf_composition_boundaries!(comp_field, comp_boundaries, 0.0)
```

### Time-Dependent Updates

For time-dependent boundaries, update during the simulation loop:

```julia
dt = 0.001  # Time step
nsteps = 10000

for timestep in 1:nsteps
    current_time = timestep * dt
    
    # Update boundaries based on current time
    update_temperature_boundaries_from_netcdf!(temp_field, temp_boundaries, timestep, dt)
    update_composition_boundaries_from_netcdf!(comp_field, comp_boundaries, timestep, dt)
    
    # Perform simulation step
    # ... (timestepping code)
end
```

## Grid Interpolation

Geodynamo.jl automatically handles grid mismatches through interpolation:

### Automatic Interpolation

```julia
# NetCDF file has 32×64 grid, simulation uses 64×128
# Interpolation happens automatically during application
temp_boundaries = load_temperature_boundaries("low_res_cmb.nc", "low_res_surface.nc")
apply_netcdf_temperature_boundaries!(temp_field, temp_boundaries)  # Auto-interpolation
```

### Manual Interpolation

```julia
# Create custom target grid
target_theta = range(0, π, length=96)
target_phi = range(0, 2π, length=193)[1:192]

# Interpolate boundary data
interpolated_data = interpolate_boundary_to_grid(temp_boundaries.inner_boundary, 
                                               target_theta, target_phi, 1)  # Time index 1
```

## Advanced Usage

### Custom Coordinate Names

The NetCDF reader supports flexible coordinate variable names:

```julia
# Automatically detects these coordinate names:
# Latitude: "theta", "colatitude", "colat", "lat", "latitude"
# Longitude: "phi", "longitude", "long", "lon"  
# Time: "time", "t", "time_index"

# Custom coordinate mapping (if needed)
custom_coords = Dict(
    "theta" => ["custom_lat", "my_theta"],
    "phi" => ["custom_lon", "my_phi"], 
    "time" => ["my_time"]
)

boundary_data = read_netcdf_boundary_data("custom_file.nc", "temperature", 
                                         coord_names=custom_coords)
```

### Field Name Auto-Detection

```julia
# Auto-detect main data variable (looks for common names)
boundary_data = read_netcdf_boundary_data("boundary_file.nc")  # Empty field name = auto-detect

# Explicitly specify field name
boundary_data = read_netcdf_boundary_data("boundary_file.nc", "my_temperature_field")
```

### Error Handling

```julia
try
    temp_boundaries = load_temperature_boundaries("cmb_temp.nc", "surface_temp.nc")
    apply_netcdf_temperature_boundaries!(temp_field, temp_boundaries)
catch e
    if isa(e, ArgumentError)
        println("Boundary validation failed: $e")
        # Handle validation errors
    elseif isa(e, BoundsError)  
        println("Time index out of range: $e")
        # Handle time indexing errors
    else
        println("Unexpected error: $e")
        rethrow()
    end
end
```

## Performance Considerations

### Memory Usage

For large time-dependent datasets:

```julia
# Check memory usage before loading
fileinfo = NCDataset("large_timedep_boundary.nc", "r") do ds
    println("Time steps: $(size(ds["time"]))")
    println("Spatial grid: $(size(ds["temperature"])[1:2])")
    println("Total data points: $(prod(size(ds["temperature"])))")
end

# Consider loading subsets for very large files
```

### Interpolation Performance

- **Nearest-neighbor interpolation**: Fast, suitable for similar grid resolutions
- **Bilinear interpolation**: More accurate but slower (planned enhancement)
- **Grid matching**: Best performance when NetCDF grid matches simulation grid

### Caching

Boundary data is cached in memory after loading. For multiple simulations:

```julia
# Load once, reuse multiple times
temp_boundaries = load_temperature_boundaries("cmb_temp.nc", "surface_temp.nc")

# Use in multiple simulations
for sim in simulations
    apply_netcdf_temperature_boundaries!(sim.temp_field, temp_boundaries)
end
```

## Troubleshooting

### Common Issues

#### File Not Found
```
ERROR: NetCDF file not found: cmb_temp.nc
```
**Solution**: Ensure file paths are correct and files exist.

#### Grid Mismatch
```
ERROR: Inner boundary nlat (32) != config nlat (64)
```
**Solution**: Either create matching grids or rely on automatic interpolation.

#### Missing Coordinates
```
WARNING: No coordinates in boundary data and size mismatch
```
**Solution**: Add coordinate variables to NetCDF file or ensure data dimensions match exactly.

#### Time Index Error
```
ERROR: time_index=15 out of range [1, 10]
```
**Solution**: Check time array length and adjust time indexing.

### Debugging

Enable detailed logging:

```julia
# Check boundary data details
boundary_stats = get_boundary_statistics(boundary_data)
for (key, val) in boundary_stats
    println("$key: $val")
end

# Verify coordinate arrays
if boundary_data.theta !== nothing
    println("Theta range: [$(minimum(boundary_data.theta)), $(maximum(boundary_data.theta))]")
end

if boundary_data.phi !== nothing
    println("Phi range: [$(minimum(boundary_data.phi)), $(maximum(boundary_data.phi))]")
end
```

## Examples and Applications

### Realistic Earth-like Boundaries

```julia
# CMB temperature with plume structures
temp_boundaries = load_temperature_boundaries("realistic_cmb.nc", "earth_surface.nc")

# Apply with realistic simulation parameters
config = create_optimized_config(128, 128, nlat=256, nlon=512)  # High resolution
domain = create_radial_domain(0.35, 1.0, 128)  # Earth-like radii

temp_field = create_shtns_temperature_field(Float64, config, domain) 
apply_netcdf_temperature_boundaries!(temp_field, temp_boundaries)
```

### Time-Varying Surface Conditions

```julia
# Seasonal or geological time-scale surface variations
surface_boundaries = load_temperature_boundaries("cmb_steady.nc", "surface_varying.nc")

# Update during simulation
for timestep in 1:simulation_steps
    geological_time = timestep * dt * 1e6  # Convert to years
    update_temperature_boundaries_from_netcdf!(temp_field, surface_boundaries, timestep, dt)
    
    # Continue simulation...
end
```

### Multi-Component Composition

```julia
# Light element release at CMB
light_element_bc = load_composition_boundaries("cmb_light_elements.nc", "surface_zero.nc")
apply_netcdf_composition_boundaries!(comp_field, light_element_bc)

# Additional compositional tracer
heavy_element_bc = load_composition_boundaries("cmb_heavy.nc", "surface_heavy.nc")
# Apply to second composition field...
```

## API Reference

### Main Functions

- `load_temperature_boundaries(inner_file, outer_file; precision=Float64)`: Load temperature boundaries
- `load_composition_boundaries(inner_file, outer_file; precision=Float64)`: Load composition boundaries
- `apply_netcdf_temperature_boundaries!(field, boundaries, time=0.0)`: Apply temperature boundaries
- `apply_netcdf_composition_boundaries!(field, boundaries, time=0.0)`: Apply composition boundaries

### Utility Functions

- `read_netcdf_boundary_data(file, field="", coord_names=default_coord_names())`: Low-level NetCDF reader
- `interpolate_boundary_to_grid(boundary_data, target_theta, target_phi, time_idx=1)`: Manual interpolation
- `get_boundary_statistics(boundary_data)`: Statistical analysis
- `print_boundary_info(boundary_set)`: Display boundary information
- `validate_netcdf_temperature_compatibility(boundaries, config)`: Validate temperature boundaries
- `validate_netcdf_composition_compatibility(boundaries, config)`: Validate composition boundaries

### Data Structures

- `BoundaryData{T}`: Single boundary data (inner or outer)
- `BoundaryConditionSet{T}`: Complete set with inner and outer boundaries

For complete API documentation, see the function docstrings in the source code.