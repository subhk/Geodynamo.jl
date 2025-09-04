---
layout: page
title: API Reference
permalink: /api-reference/
nav_order: 3
description: "Complete function and type documentation for Geodynamo.jl"
---

# API Reference

Complete function and type documentation for Geodynamo.jl

---

## Core Configuration

### `create_optimized_config`

```julia
create_optimized_config(lmax, mmax; kwargs...)
```

Create an optimized SHTns configuration for geodynamo simulations.

**Parameters:**
- `lmax::Int` - Maximum spherical harmonic degree
- `mmax::Int` - Maximum spherical harmonic order  
- `nlat::Int` - Number of latitude points (default: lmax+2)
- `nlon::Int` - Number of longitude points (default: max(2*lmax+1, 4))
- `use_threading::Bool` - Enable CPU threading (default: true)
- `use_simd::Bool` - Enable SIMD vectorization (default: true)
- `norm::Symbol` - Normalization scheme (default: :orthonormal)

**Returns:** `SHTnsKitConfig`

**Example:**
```julia
config = create_optimized_config(64, 64, nlat=128, nlon=256, use_threading=true)
```

---

### `create_radial_domain`

```julia
create_radial_domain(inner_radius, outer_radius, nr)
```

Create a radial domain for spherical shell geometry.

**Parameters:**
- `inner_radius::Float64` - Inner radius (normalized, typically 0.35 for Earth)
- `outer_radius::Float64` - Outer radius (normalized, typically 1.0)  
- `nr::Int` - Number of radial grid points

**Returns:** `RadialDomain`

**Example:**
```julia
domain = create_radial_domain(0.35, 1.0, 64)  # Earth-like core
```

---

## Field Creation

### Temperature Fields

```julia
create_shtns_temperature_field(T, config, domain)
```

Create a temperature field for thermal convection.

**Parameters:**
- `T::Type` - Precision type (Float32 or Float64)
- `config::SHTnsKitConfig` - SHTns configuration
- `domain::RadialDomain` - Radial domain

**Returns:** `SHTnsTemperatureField{T}`

---

### Composition Fields

```julia
create_shtns_composition_field(T, config, domain)
```

Create a compositional field for chemical convection.

**Parameters:**
- `T::Type` - Precision type (Float32 or Float64)
- `config::SHTnsKitConfig` - SHTns configuration
- `domain::RadialDomain` - Radial domain

**Returns:** `SHTnsCompositionField{T}`

---

### Velocity Fields

```julia
create_shtns_velocity_fields(T, config, domain)
```

Create velocity fields for fluid motion.

**Parameters:**
- `T::Type` - Precision type (Float32 or Float64)
- `config::SHTnsKitConfig` - SHTns configuration  
- `domain::RadialDomain` - Radial domain

**Returns:** `SHTnsVelocityFields{T}`

---

### Magnetic Fields

```julia
create_shtns_magnetic_fields(T, config, domain_oc, domain_ic)
```

Create magnetic fields for dynamo simulation.

**Parameters:**
- `T::Type` - Precision type (Float32 or Float64)
- `config::SHTnsKitConfig` - SHTns configuration
- `domain_oc::RadialDomain` - Outer core domain
- `domain_ic::RadialDomain` - Inner core domain

**Returns:** `SHTnsMagneticFields{T}`

---

## Boundary Conditions

### NetCDF Boundaries

#### Temperature Boundaries

```julia
load_temperature_boundaries(inner_file, outer_file; precision=Float64)
```

Load temperature boundary conditions from NetCDF files.

**Parameters:**
- `inner_file::String` - Path to inner boundary NetCDF file
- `outer_file::String` - Path to outer boundary NetCDF file  
- `precision::Type` - Data precision (Float32 or Float64)

**Returns:** `BoundaryConditionSet{T}`

---

#### Composition Boundaries

```julia
load_composition_boundaries(inner_file, outer_file; precision=Float64)
```

Load compositional boundary conditions from NetCDF files.

**Parameters:**
- `inner_file::String` - Path to inner boundary NetCDF file
- `outer_file::String` - Path to outer boundary NetCDF file
- `precision::Type` - Data precision (Float32 or Float64)

**Returns:** `BoundaryConditionSet{T}`

---

### Hybrid Boundaries

#### Temperature Hybrid Boundaries

```julia
create_hybrid_temperature_boundaries(inner_spec, outer_spec, config; precision=Float64)
```

Create hybrid temperature boundaries mixing NetCDF files and programmatic patterns.

**Parameters:**
- `inner_spec` - Inner boundary specification
- `outer_spec` - Outer boundary specification  
- `config::SHTnsKitConfig` - SHTns configuration
- `precision::Type` - Data precision

**Boundary Specifications:**
- `"file.nc"` - Load from NetCDF file
- `(:pattern, amplitude, parameters)` - Programmatic pattern

**Returns:** `BoundaryConditionSet{T}`

---

#### Composition Hybrid Boundaries

```julia
create_hybrid_composition_boundaries(inner_spec, outer_spec, config; precision=Float64)
```

Similar to temperature boundaries but for compositional fields.

---

### Programmatic Patterns

| Pattern | Syntax | Description |
|---------|--------|-------------|
| `:uniform` | `(:uniform, amplitude)` | Constant value |
| `:y11` | `(:y11, base, Dict("amplitude"=>amp))` | Y₁₁ spherical harmonic |
| `:plume` | `(:plume, base, Dict("width"=>w, "center_theta"=>θ))` | Gaussian plume |
| `:hemisphere` | `(:hemisphere, amp, Dict("axis"=>"z"))` | Half-sphere pattern |
| `:custom` | `(:custom, amp, Dict("function"=>func))` | Custom function |

**Example:**
```julia
boundaries = create_hybrid_temperature_boundaries(
    (:plume, 4200.0, Dict(
        "width" => π/6, 
        "center_theta" => π/3,
        "center_phi" => 0.0
    )),
    "surface_temps.nc",
    config
)
```

---

## Application Functions

### Apply Boundaries

```julia
apply_netcdf_temperature_boundaries!(temp_field, boundary_set, current_time=0.0)
```

Apply temperature boundary conditions to a field.

**Parameters:**
- `temp_field::SHTnsTemperatureField` - Temperature field to modify
- `boundary_set::BoundaryConditionSet` - Boundary conditions
- `current_time::Float64` - Current simulation time

---

```julia
apply_netcdf_composition_boundaries!(comp_field, boundary_set, current_time=0.0)
```

Apply compositional boundary conditions to a field.

---

### Time Evolution

```julia
update_temperature_boundaries_from_netcdf!(temp_field, boundary_set, timestep, dt)
```

Update temperature boundaries during time evolution.

**Parameters:**
- `temp_field::SHTnsTemperatureField` - Temperature field
- `boundary_set::BoundaryConditionSet` - Boundary conditions
- `timestep::Int` - Current time step number
- `dt::Float64` - Time step size

---

## Simulation Functions

### Nonlinear Terms

```julia
compute_temperature_nonlinear!(temp_field, vel_field; geometry=:shell)
```

Compute nonlinear terms for temperature evolution.

**Parameters:**
- `temp_field::SHTnsTemperatureField` - Temperature field
- `vel_field::SHTnsVelocityFields` - Velocity field
- `geometry::Symbol` - Geometry type (:shell or :ball)

---

```julia
compute_magnetic_nonlinear!(mag_fields, vel_fields, rotation_rate; geometry=:shell)
```

Compute nonlinear terms for magnetic field evolution.

**Parameters:**
- `mag_fields::SHTnsMagneticFields` - Magnetic field structure
- `vel_fields::SHTnsVelocityFields` - Velocity fields  
- `rotation_rate::Float64` - Rotation rate

---

## Utility Functions

### Information

```julia
print_boundary_info(boundary_set)
```

Print detailed boundary information.

---

```julia
get_boundary_statistics(boundary_data) -> Dict
```

Get statistical summary with keys: "min", "max", "mean", "std", "nlat", "nlon"

---

### Validation

```julia
validate_netcdf_temperature_compatibility(boundary_set, config) -> Bool
```

Check compatibility between boundaries and configuration.

---

### Performance Monitoring

```julia
reset_performance_stats!()
```

Reset performance counters.

---

```julia
print_performance_report()
```

Display detailed performance statistics.

---

```julia
@timed_transform expr
```

Time transform operations and update statistics.

**Example:**
```julia
@timed_transform begin
    for step in 1:nsteps
        compute_temperature_nonlinear!(temp_field, vel_field)
    end
end
print_performance_report()
```

---

## Data Structures

### Core Types

#### `BoundaryData{T}`

Single boundary condition data.

**Fields:**
- `data::Array{T,2}` - Values (nlat × nlon)
- `theta::Vector{Float64}` - Colatitudes [0,π]
- `phi::Vector{Float64}` - Longitudes [0,2π]
- `metadata::Dict` - Additional information

---

#### `BoundaryConditionSet{T}`

Complete boundary set (inner + outer).

**Fields:**
- `inner_boundary::BoundaryData{T}` - Inner conditions
- `outer_boundary::BoundaryData{T}` - Outer conditions
- `field_type::String` - "temperature" or "composition"

---

#### `SHTnsKitConfig`

SHTns configuration.

**Fields:**
- `lmax::Int` - Maximum degree
- `mmax::Int` - Maximum order  
- `nlat::Int` - Latitude points
- `nlon::Int` - Longitude points
- `l_values::Vector{Int}` - Degrees
- `m_values::Vector{Int}` - Orders

---

#### Field Structures

**`SHTnsTemperatureField{T}`**
- `temperature::SHTnsScalarField{T}` - Physical space
- `spectral::SHTnsSpectralField{T}` - Spectral space
- `nonlinear::SHTnsSpectralField{T}` - Nonlinear terms

**`SHTnsMagneticFields{T}`**
- `magnetic::SHTnsVectorField{T}` - Physical magnetic field
- `toroidal::SHTnsSpectralField{T}` - Toroidal component
- `poloidal::SHTnsSpectralField{T}` - Poloidal component

---

## Error Types

### `BoundaryCompatibilityError`
Incompatible boundary conditions and configuration.

### `NetCDFFormatError`  
Invalid NetCDF file format or missing variables.

### `ConfigurationError`
Invalid SHTns configuration parameters.

---

## Constants

```julia
# Physical constants
const R_EARTH = 6.371e6        # Earth radius (m)
const G_GRAV = 9.81           # Gravity (m/s²)
const OMEGA_EARTH = 7.27e-5   # Earth rotation (rad/s)

# Default parameters
const DEFAULT_LMAX = 32       # Default max degree
const DEFAULT_PRECISION = Float64
```

---

## Advanced Usage

### Custom Pattern Functions

```julia
function my_custom_pattern(theta::Float64, phi::Float64)::Float64
    # theta: colatitude [0, π]
    # phi: longitude [0, 2π]
    return sin(3*theta) * cos(2*phi) + 0.5*cos(theta)^2
end

boundaries = create_hybrid_temperature_boundaries(
    (:custom, 4000.0, Dict("function" => my_custom_pattern)),
    (:uniform, 300.0),
    config
)
```

### MPI Integration

```julia
using MPI
MPI.Init()

# All operations are automatically MPI-aware
config = create_optimized_config(128, 128, nlat=256, nlon=512)
```

---

## See Also

- [Getting Started Guide](getting-started.html) - Comprehensive tutorial
- [Visualization Guide](visualization.html) - Plotting tools
- [Examples](examples.html) - Working code examples
- [GitHub Repository](https://github.com/subhk/Geodynamo.jl) - Source and issues