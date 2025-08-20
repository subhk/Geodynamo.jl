# Getting Started with Geodynamo.jl

Welcome to Geodynamo.jl! This guide will help you get up and running with geodynamo simulations using spherical harmonic transforms and flexible boundary conditions.

## Quick Start

### Installation

```julia
using Pkg
Pkg.add("Geodynamo")
```

### Your First Simulation

```julia
using Geodynamo

# 1. Create a basic configuration
config = create_optimized_config(32, 32, nlat=64, nlon=128)

# 2. Set up the simulation domain  
domain = create_radial_domain(0.35, 1.0, 64)  # Inner radius, outer radius, radial points

# 3. Create temperature field
temp_field = create_shtns_temperature_field(Float64, config, domain)

# 4. Set simple boundary conditions
apply_netcdf_temperature_boundaries!(temp_field, 
    create_hybrid_temperature_boundaries(
        (:uniform, 4000.0),  # Hot CMB
        (:uniform, 300.0),   # Cool surface
        config
    )
)

println("Your first geodynamo simulation is ready!")
```

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Setting Up Simulations](#setting-up-simulations)
3. [Boundary Conditions](#boundary-conditions)
4. [Running Simulations](#running-simulations)
5. [Examples](#examples)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Topics](#advanced-topics)

---

## Core Concepts

### What is Geodynamo.jl?

Geodynamo.jl is a Julia package for simulating Earth's magnetic field generation through the geodynamo process. It uses:

- **Spherical Harmonic Transforms (SHTns)**: Efficient spectral methods for spherical geometry
- **Flexible Boundary Conditions**: Support for NetCDF data files and programmatic patterns
- **High Performance**: CPU-optimized with SIMD vectorization and threading
- **MPI Parallelization**: Scalable to large clusters

### Key Components

- **Temperature Field**: Controls buoyancy and convection
- **Compositional Field**: Tracks light/heavy element distribution
- **Magnetic Field**: The dynamo-generated magnetic field
- **Velocity Field**: Fluid motion driving the dynamo

### Coordinate System

- **Spherical coordinates**: (r, θ, φ) where θ is colatitude [0,π], φ is longitude [0,2π]
- **Radial domain**: From inner core boundary to core-mantle boundary
- **Spectral representation**: Spherical harmonics Y_l^m for angular dependence

---

## Setting Up Simulations

### Step 1: Configuration

Create an SHTns configuration specifying resolution and optimization:

```julia
using Geodynamo

# Basic configuration
config = create_optimized_config(
    32, 32,              # lmax, mmax (spherical harmonic resolution)
    nlat=64, nlon=128,   # Physical grid resolution
    use_threading=true,   # Enable CPU threading
    use_simd=true        # Enable SIMD vectorization
)
```

#### Resolution Guidelines

| Problem Size | lmax/mmax | nlat × nlon | Use Case |
|-------------|-----------|-------------|----------|
| Learning/Testing | 16-32 | 32 × 64 | Quick experiments |
| Research | 64-128 | 128 × 256 | Production simulations |
| High-Resolution | 256+ | 512 × 1024 | Publication quality |

### Step 2: Domain Setup

Define the radial domain (spherical shell):

```julia
# Earth-like parameters
inner_radius = 0.35    # Inner core boundary (normalized)
outer_radius = 1.0     # Core-mantle boundary (normalized)
nr = 64                # Number of radial grid points

domain = create_radial_domain(inner_radius, outer_radius, nr)
```

### Step 3: Create Fields

Initialize the physical fields for your simulation:

```julia
# Temperature field (drives convection)
temp_field = create_shtns_temperature_field(Float64, config, domain)

# Compositional field (optional, for chemical convection)
comp_field = create_shtns_composition_field(Float64, config, domain)

# Velocity field (fluid motion)
vel_field = create_shtns_velocity_fields(Float64, config, domain)

# Magnetic field (dynamo output)
mag_field = create_shtns_magnetic_fields(Float64, config, domain)
```

---

## Boundary Conditions

Geodynamo.jl offers flexible boundary condition specification. You can mix and match different approaches:

### Option 1: Simple Programmatic Boundaries

Perfect for learning and testing:

```julia
# Create uniform temperature boundaries
temp_boundaries = create_hybrid_temperature_boundaries(
    (:uniform, 4000.0),    # Inner boundary: 4000 K (hot CMB)
    (:uniform, 300.0),     # Outer boundary: 300 K (cool surface)
    config
)

# Apply to temperature field
apply_netcdf_temperature_boundaries!(temp_field, temp_boundaries)
```

### Option 2: Spherical Harmonic Patterns

Add realistic spatial structure:

```julia
# CMB with Y₁₁ perturbation, uniform surface
temp_boundaries = create_hybrid_temperature_boundaries(
    (:y11, 4000.0, Dict("amplitude" => 200.0)),  # CMB: base + Y₁₁ pattern
    (:uniform, 300.0),                           # Surface: uniform
    config
)
```

### Option 3: Realistic Patterns

Use physical patterns like plumes:

```julia
# Hot plume at CMB, uniform surface
temp_boundaries = create_hybrid_temperature_boundaries(
    (:plume, 4200.0, Dict(
        "center_theta" => π/3,    # Plume location (colatitude)
        "center_phi" => π/4,      # Plume location (longitude)  
        "width" => π/8            # Plume width
    )),
    (:uniform, 300.0),
    config
)
```

### Option 4: NetCDF Data Files

Use observational or numerical model data:

```julia
# First create sample files (one-time setup)
include("examples/create_sample_netcdf_boundaries.jl")

# Load from NetCDF files
temp_boundaries = load_temperature_boundaries("cmb_temp.nc", "surface_temp.nc")
apply_netcdf_temperature_boundaries!(temp_field, temp_boundaries)
```

### Option 5: Hybrid (Mix NetCDF + Programmatic)

Combine data files with analytical patterns:

```julia
# NetCDF CMB data + simple uniform surface
temp_boundaries = create_hybrid_temperature_boundaries(
    "realistic_cmb.nc",       # Complex CMB from data file
    (:uniform, 300.0),        # Simple uniform surface
    config
)
```

### Available Programmatic Patterns

| Pattern | Description | Parameters |
|---------|-------------|------------|
| `:uniform` | Constant value | amplitude |
| `:y11` | Y₁₁ spherical harmonic | amplitude |
| `:y20` | Y₂₀ zonal harmonic | amplitude |
| `:plume` | Gaussian plume | center_theta, center_phi, width |
| `:hemisphere` | Half-sphere pattern | axis ("x", "y", "z") |
| `:dipole` | Dipolar (Y₁₀) | amplitude |
| `:checkerboard` | Alternating blocks | nblocks_theta, nblocks_phi |
| `:custom` | Your function | function |

---

## Running Simulations

### Basic Time Stepping

```julia
# Simulation parameters
dt = 0.001        # Time step
nsteps = 1000     # Number of steps
output_freq = 100 # Output every N steps

# Time stepping loop
for step in 1:nsteps
    current_time = step * dt
    
    # Update time-dependent boundaries (if applicable)
    update_temperature_boundaries_from_netcdf!(temp_field, temp_boundaries, step, dt)
    
    # Compute nonlinear terms
    compute_temperature_nonlinear!(temp_field, vel_field)
    
    # Time step (simplified - actual implementation more complex)
    # solve_implicit_step!(temp_field, dt)
    
    # Output results
    if step % output_freq == 0
        println("Step $step, time = $(current_time)")
        # write_fields!(...) # Save to files
    end
end
```

### Monitoring Performance

Track simulation performance:

```julia
# Reset performance counters
reset_performance_stats!()

# Run simulation with monitoring
@timed_transform begin
    for step in 1:nsteps
        # ... simulation steps
    end
end

# View performance report
print_performance_report()
```

---

## Examples

### Example 1: Basic Thermal Convection

```julia
using Geodynamo

function basic_thermal_convection()
    println("Basic Thermal Convection Example")
    
    # Setup
    config = create_optimized_config(32, 32, nlat=64, nlon=128)
    domain = create_radial_domain(0.35, 1.0, 64)
    
    # Create fields
    temp_field = create_shtns_temperature_field(Float64, config, domain)
    
    # Simple boundaries: hot bottom, cold top
    temp_boundaries = create_hybrid_temperature_boundaries(
        (:uniform, 4000.0),  # Hot CMB
        (:uniform, 300.0),   # Cool surface
        config
    )
    
    # Apply boundaries
    apply_netcdf_temperature_boundaries!(temp_field, temp_boundaries)
    
    # Add small perturbation to initiate convection
    set_temperature_ic!(temp_field, :random_perturbation, domain)
    
    println("Basic thermal convection setup complete")
    print_boundary_info(temp_boundaries)
end

basic_thermal_convection()
```

### Example 2: Plume-Driven Convection

```julia
function plume_convection()
    println("Plume-Driven Convection Example")
    
    config = create_optimized_config(64, 64, nlat=128, nlon=256)
    domain = create_radial_domain(0.35, 1.0, 64)
    
    temp_field = create_shtns_temperature_field(Float64, config, domain)
    
    # Hot plume at CMB
    temp_boundaries = create_hybrid_temperature_boundaries(
        (:plume, 4200.0, Dict(
            "center_theta" => π/3,    # 60° from north pole
            "center_phi" => 0.0,      # Prime meridian
            "width" => π/6            # 30° width
        )),
        (:uniform, 300.0),            # Uniform cool surface
        config
    )
    
    apply_netcdf_temperature_boundaries!(temp_field, temp_boundaries)
    
    println("Plume convection setup complete")
    
    # Show boundary statistics
    inner_stats = get_boundary_statistics(temp_boundaries.inner_boundary)
    println("CMB temperature range: [$(round(inner_stats["min"])), $(round(inner_stats["max"]))] K")
end

plume_convection()
```

### Example 3: Using Real Data

```julia
function realistic_boundaries()
    println("Realistic Boundary Conditions Example")
    
    # Create sample NetCDF files if needed
    if !isfile("cmb_temp.nc")
        include("examples/create_sample_netcdf_boundaries.jl")
    end
    
    config = create_optimized_config(64, 64, nlat=128, nlon=256)
    domain = create_radial_domain(0.35, 1.0, 64)
    
    # Load temperature boundaries from NetCDF files
    temp_boundaries = load_temperature_boundaries("cmb_temp.nc", "surface_temp.nc")
    
    # Load composition boundaries  
    comp_boundaries = load_composition_boundaries("cmb_composition.nc", "surface_composition.nc")
    
    # Create fields
    temp_field = create_shtns_temperature_field(Float64, config, domain)
    comp_field = create_shtns_composition_field(Float64, config, domain)
    
    # Apply boundaries
    apply_netcdf_temperature_boundaries!(temp_field, temp_boundaries)
    apply_netcdf_composition_boundaries!(comp_field, comp_boundaries)
    
    println("Realistic boundaries loaded")
    print_boundary_info(temp_boundaries)
    print_boundary_info(comp_boundaries)
end

realistic_boundaries()
```

### Example 4: Time-Dependent Boundaries

```julia
function time_dependent_example()
    println("Time-Dependent Boundaries Example")
    
    config = create_optimized_config(32, 32, nlat=64, nlon=128)
    domain = create_radial_domain(0.35, 1.0, 64)
    
    # Create rotating plume pattern
    rotating_inner = create_time_dependent_programmatic_boundary(
        :plume, config, (0.0, 10.0), 50,  # 50 time steps over 10 time units
        amplitude=4200.0,
        parameters=Dict(
            "width" => π/6,
            "center_theta" => π/3,
            "time_factor" => 2π,     # One full rotation
        )
    )
    
    # Create boundary set
    temp_boundaries = BoundaryConditionSet(
        rotating_inner,
        create_programmatic_boundary(:uniform, config, amplitude=300.0),
        "temperature",
        time()
    )
    
    temp_field = create_shtns_temperature_field(Float64, config, domain)
    
    # Simulate time evolution
    dt = 0.2
    for step in 1:10
        current_time = step * dt
        apply_netcdf_temperature_boundaries!(temp_field, temp_boundaries, current_time)
        println("Step $step: Applied boundaries at time $current_time")
    end
    
    println("Time-dependent boundary example complete")
end

time_dependent_example()
```

---

## Troubleshooting

### Common Issues and Solutions

#### Issue: "NetCDF file not found"
```
ERROR: NetCDF file not found: cmb_temp.nc
```
**Solution**: Create sample files first:
```julia
include("examples/create_sample_netcdf_boundaries.jl")
```

#### Issue: "Grid size mismatch"
```
ERROR: Inner boundary nlat (32) != config nlat (64)
```
**Solution**: Either:
1. Match grid sizes: `create_optimized_config(32, 32, nlat=32, nlon=64)`
2. Or rely on automatic interpolation (may be slower)

#### Issue: "SHTnsKit not found"
```
ERROR: Package SHTnsKit not found
```
**Solution**: Install local dependency:
```julia
using Pkg
Pkg.develop(path="../SHTnsKit.jl")  # Adjust path as needed
```

#### Issue: Performance warnings
```
WARNING: Type instability detected
```
**Solution**: Check field types and use consistent precision:
```julia
temp_field = create_shtns_temperature_field(Float64, config, domain)  # Explicit Float64
```

### Debugging Tips

1. **Start Simple**: Begin with low resolution and uniform boundaries
2. **Check Boundaries**: Use `print_boundary_info()` to inspect loaded data
3. **Monitor Performance**: Use `@timed_transform` and `print_performance_report()`
4. **Validate Data**: Use `get_boundary_statistics()` to check loaded values

### Getting Help

- **Documentation**: Check the docs/ directory for detailed guides
- **Examples**: Run examples/ scripts to see working code
- **Issues**: Report bugs on the GitHub repository
- **Community**: Ask questions in discussions

---

## Advanced Topics

### MPI Parallelization

For large simulations, use MPI:

```bash
mpirun -np 4 julia --project=. my_simulation.jl
```

```julia
using MPI
MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)

if rank == 0
    println("Running on $nprocs processes")
end

# ... rest of simulation
```

### Custom Boundary Patterns

Create your own boundary patterns:

```julia
# Define custom function
function my_boundary_pattern(theta, phi)
    # Your mathematical expression here
    return sin(3*theta) * cos(2*phi) + 0.5*cos(theta)^2
end

# Use in boundary specification
boundaries = create_hybrid_temperature_boundaries(
    (:custom, 4000.0, Dict("function" => my_boundary_pattern)),
    (:uniform, 300.0),
    config
)
```

### High-Performance Tips

1. **Use appropriate precision**: Float64 for accuracy, Float32 for speed
2. **Enable optimizations**: `use_threading=true`, `use_simd=true`
3. **Choose resolution wisely**: Balance accuracy vs. computational cost
4. **Profile your code**: Use `@timed_transform` to identify bottlenecks

### Creating NetCDF Files

For your own data, create NetCDF files with this structure:

```julia
using NCDatasets

NCDataset("my_boundary.nc", "c") do ds
    # Define dimensions
    defDim(ds, "lat", nlat)
    defDim(ds, "lon", nlon)
    
    # Define coordinates
    defVar(ds, "theta", Float64, ("lat",))  # Colatitude [0, π]
    defVar(ds, "phi", Float64, ("lon",))    # Longitude [0, 2π]
    
    # Define data variable
    defVar(ds, "temperature", Float64, ("lat", "lon"), 
           attrib=Dict("units" => "K", "long_name" => "Temperature"))
    
    # Write data
    ds["theta"][:] = your_theta_data
    ds["phi"][:] = your_phi_data  
    ds["temperature"][:] = your_temperature_data
end
```

---

## What's Next?

1. **Run the Examples**: Start with the provided example scripts
2. **Modify Parameters**: Experiment with different resolutions and boundary conditions
3. **Add Physics**: Include magnetic field generation and compositional convection
4. **Scale Up**: Move to higher resolution and MPI parallelization
5. **Analyze Results**: Use built-in analysis tools for scientific insights

### Learning Path

1. **Beginner**: Start with uniform boundaries and low resolution
2. **Intermediate**: Use NetCDF files and spherical harmonic patterns  
3. **Advanced**: Create custom patterns and time-dependent boundaries
4. **Expert**: Develop new physics modules and optimization techniques

---

## Quick Reference Card

### Essential Commands

```julia
# Configuration
config = create_optimized_config(lmax, mmax, nlat=nlat, nlon=nlon)
domain = create_radial_domain(ri, ro, nr)

# Fields  
temp_field = create_shtns_temperature_field(Float64, config, domain)

# Boundaries - Programmatic
boundaries = create_hybrid_temperature_boundaries((:pattern, amplitude), (:pattern, amplitude), config)

# Boundaries - NetCDF
boundaries = load_temperature_boundaries("inner.nc", "outer.nc")

# Apply boundaries
apply_netcdf_temperature_boundaries!(temp_field, boundaries)

# Performance monitoring
reset_performance_stats!()
@timed_transform begin
    # simulation code
end
print_performance_report()
```

### Pattern Quick Reference

| Pattern | Example | Description |
|---------|---------|-------------|
| `:uniform` | `(:uniform, 300.0)` | Constant value |
| `:y11` | `(:y11, 4000.0, Dict("amplitude"=>200))` | Y₁₁ + perturbation |
| `:plume` | `(:plume, 4200.0, Dict("width"=>π/6))` | Gaussian plume |
| `:dipole` | `(:dipole, 1000.0)` | Dipolar pattern |

Welcome to geodynamo modeling!