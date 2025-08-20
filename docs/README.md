# Geodynamo.jl Documentation

**Earth's Magnetic Field Simulation in Julia**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Julia](https://img.shields.io/badge/Julia-1.6%2B-blueviolet)](https://julialang.org/)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://github.com/subhk/Geodynamo.jl)

High-performance geodynamo modeling with spherical harmonic transforms and flexible boundary conditions.

---

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
temp_boundaries = create_hybrid_temperature_boundaries(
    (:uniform, 4000.0),  # Hot CMB
    (:uniform, 300.0),   # Cool surface
    config
)
apply_netcdf_temperature_boundaries!(temp_field, temp_boundaries)

println("✓ Your first geodynamo simulation is ready!")
```

---

## Key Features

| Feature | Description |
|---------|-------------|
| **🌊 Spherical Harmonic Transforms** | Efficient spectral methods using SHTns for spherical geometry |
| **Flexible Boundary Conditions** | NetCDF files, programmatic patterns, and hybrid approaches |
| **High Performance** | CPU-optimized with SIMD vectorization and MPI parallelization |
| **NetCDF Integration** | Read observational data and numerical model outputs |
| **Performance Monitoring** | Built-in profiling and optimization tools |
| **Time-Dependent Boundaries** | Evolving boundary conditions during simulation |

---

## 📚 Table of Contents

- [Core Concepts](#-core-concepts)
- [Setting Up Simulations](#-setting-up-simulations)
- [Boundary Conditions](#-boundary-conditions)
- [Running Simulations](#-running-simulations)
- [Examples](#-examples)
- [Performance & Optimization](#-performance--optimization)
- [Troubleshooting](#-troubleshooting)
- [API Reference](#-api-reference)

---

## Core Concepts

### What is Geodynamo.jl?

Geodynamo.jl simulates Earth's magnetic field generation through the geodynamo process using:

- **Spherical Harmonic Transforms (SHTns)**: Efficient spectral methods for spherical geometry
- **Flexible Boundary Conditions**: Support for NetCDF data files and programmatic patterns
- **High Performance**: CPU-optimized with SIMD vectorization and threading
- **MPI Parallelization**: Scalable to large computational clusters

### Key Components

| Component | Description |
|-----------|-------------|
| **Temperature Field** | Controls buoyancy and thermal convection |
| **Compositional Field** | Tracks light/heavy element distribution |
| **Magnetic Field** | The dynamo-generated magnetic field |
| **Velocity Field** | Fluid motion driving the dynamo |

### Coordinate System

- **Spherical coordinates**: `(r, θ, φ)` where `θ` is colatitude `[0,π]`, `φ` is longitude `[0,2π]`
- **Radial domain**: From inner core boundary to core-mantle boundary
- **Spectral representation**: Spherical harmonics `Y_l^m` for angular dependence

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

| Problem Size | lmax/mmax | nlat × nlon | Use Case | Memory |
|-------------|-----------|-------------|----------|---------|
| **Learning/Testing** | 16-32 | 32 × 64 | Quick experiments | ~100 MB |
| **Research** | 64-128 | 128 × 256 | Production simulations | ~1-4 GB |
| **High-Resolution** | 256+ | 512 × 1024 | Publication quality | ~16+ GB |

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

Geodynamo.jl offers **five different approaches** to specify boundary conditions:

### 🔹 Option 1: Simple Programmatic Boundaries

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

### 🔹 Option 2: Spherical Harmonic Patterns

Add realistic spatial structure:

```julia
# CMB with Y₁₁ perturbation, uniform surface
temp_boundaries = create_hybrid_temperature_boundaries(
    (:y11, 4000.0, Dict("amplitude" => 200.0)),  # CMB: base + Y₁₁ pattern
    (:uniform, 300.0),                           # Surface: uniform
    config
)
```

### 🔹 Option 3: Realistic Physical Patterns

Use physical patterns like plumes and hemispheres:

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

### 🔹 Option 4: NetCDF Data Files

Use observational or numerical model data:

```julia
# First create sample files (one-time setup)
include("examples/create_sample_netcdf_boundaries.jl")

# Load from NetCDF files
temp_boundaries = load_temperature_boundaries("cmb_temp.nc", "surface_temp.nc")
apply_netcdf_temperature_boundaries!(temp_field, temp_boundaries)

# Inspect loaded data
print_boundary_info(temp_boundaries)
```

### 🔹 Option 5: Hybrid (Mix NetCDF + Programmatic)

**Most Flexible**: Combine data files with analytical patterns:

```julia
# NetCDF CMB data + simple uniform surface
temp_boundaries = create_hybrid_temperature_boundaries(
    "realistic_cmb.nc",       # Complex CMB from data file
    (:uniform, 300.0),        # Simple uniform surface
    config
)

# Programmatic inner + NetCDF outer
temp_boundaries = create_hybrid_temperature_boundaries(
    (:plume, 4200.0, Dict("width" => π/6)),  # Analytical plume pattern
    "earth_surface_temps.nc",                # Real surface temperature data
    config
)
```

### Available Programmatic Patterns

| Pattern | Syntax | Description | Parameters |
|---------|--------|-------------|------------|
| `:uniform` | `(:uniform, 300.0)` | Constant value | amplitude |
| `:y11` | `(:y11, 4000.0, Dict("amplitude"=>200))` | Y₁₁ spherical harmonic | amplitude |
| `:y20` | `(:y20, 1000.0)` | Y₂₀ zonal harmonic | amplitude |
| `:y22` | `(:y22, 500.0)` | Y₂₂ sectoral harmonic | amplitude |
| `:plume` | `(:plume, 4200.0, Dict("width"=>π/6))` | Gaussian plume | center_theta, center_phi, width |
| `:hemisphere` | `(:hemisphere, 1000.0, Dict("axis"=>"z"))` | Half-sphere pattern | axis ("x", "y", "z") |
| `:dipole` | `(:dipole, 1000.0)` | Dipolar (Y₁₀) pattern | amplitude |
| `:quadrupole` | `(:quadrupole, 800.0)` | Quadrupolar pattern | amplitude |
| `:checkerboard` | `(:checkerboard, 500.0, Dict("nblocks_theta"=>4))` | Alternating blocks | nblocks_theta, nblocks_phi |
| `:custom` | `(:custom, 100.0, Dict("function"=>my_func))` | Your own function | function |

### Custom Function Example

```julia
# Define your own boundary pattern
my_custom_pattern(theta, phi) = sin(3*theta) * cos(2*phi) + 0.5*cos(theta)^2

temp_boundaries = create_hybrid_temperature_boundaries(
    "realistic_cmb.nc",                                    # Inner: from NetCDF
    (:custom, 350.0, Dict("function" => my_custom_pattern)), # Outer: your function
    config
)
```

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

### Performance Monitoring

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

**Example Output:**
```
╔══════════════════════════════════════════════════════════════╗
║                    Transform Performance Report              ║
╠══════════════════════════════════════════════════════════════╣
║ Total Transforms:        1250                                ║
║ Total Time:             12.456 s                             ║
║ Average Time:            9.965 ms                            ║
║ CPU Transforms:          1250                                ║
║ Memory Allocated:        245.3 MB                            ║
║ Communication Time:       15.2%                              ║
╚══════════════════════════════════════════════════════════════╝
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
    
    println("✓ Basic thermal convection setup complete")
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
    
    println("✓ Plume convection setup complete")
    
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
    
    println("✓ Realistic boundaries loaded")
    print_boundary_info(temp_boundaries)
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
    
    println("✓ Time-dependent boundary example complete")
end

time_dependent_example()
```

### Example 5: Hybrid Approach (NetCDF + Programmatic)

```julia
function hybrid_boundaries_example()
    println("Hybrid Boundaries Example")
    
    config = create_optimized_config(64, 64, nlat=128, nlon=256)
    domain = create_radial_domain(0.35, 1.0, 64)
    
    # Temperature: NetCDF inner + programmatic outer
    temp_boundaries = create_hybrid_temperature_boundaries(
        "cmb_temp.nc",        # Complex CMB from data file
        (:uniform, 300.0),    # Simple uniform surface
        config
    )
    
    # Composition: programmatic inner + NetCDF outer
    comp_boundaries = create_hybrid_composition_boundaries(
        (:plume, 0.8, Dict(
            "center_theta" => π/4,
            "width" => π/8
        )),                           # Light element plume at CMB
        "surface_composition.nc",     # Surface composition from data
        config
    )
    
    # Create fields and apply boundaries
    temp_field = create_shtns_temperature_field(Float64, config, domain)
    comp_field = create_shtns_composition_field(Float64, config, domain)
    
    apply_netcdf_temperature_boundaries!(temp_field, temp_boundaries)
    apply_netcdf_composition_boundaries!(comp_field, comp_boundaries)
    
    println("✓ Hybrid boundaries example complete")
    print_boundary_info(temp_boundaries)
    print_boundary_info(comp_boundaries)
end

hybrid_boundaries_example()
```

---

## Performance & Optimization

### CPU Optimizations

| Optimization | Speedup | Description |
|-------------|---------|-------------|
| **SIMD Vectorization** | 20-30% | Enhanced vector operations with `@simd ivdep` |
| **Multi-threading** | 35-50% | Optimal thread utilization across CPU cores |
| **Memory Pooling** | 30-40% | Reduced memory allocations through reuse |
| **Type Stability** | 15-25% | Eliminates dynamic dispatch overhead |
| **Thread-local Caching** | 25-40% | Faster transforms under threading |

### MPI Parallelization

```julia
using MPI
MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)

if rank == 0
    println("Running on $nprocs processes")
end

# Create MPI-aware configuration
config = create_optimized_config(128, 128, nlat=256, nlon=512)

# Simulation continues with automatic MPI parallelization...
```

### Performance Monitoring

```julia
# Reset counters
reset_performance_stats!()

# Monitor simulation
@timed_transform begin
    for step in 1:nsteps
        compute_temperature_nonlinear!(temp_field, vel_field)
        solve_implicit_step!(temp_field, dt)
    end
end

# View detailed report
print_performance_report()
```

### High-Performance Tips

1. **Use appropriate precision**: `Float64` for accuracy, `Float32` for speed
2. **Enable all optimizations**: `use_threading=true`, `use_simd=true`
3. **Choose resolution wisely**: Balance accuracy vs. computational cost
4. **Profile your code**: Use `@timed_transform` to identify bottlenecks
5. **Match grid sizes**: NetCDF grids matching simulation grids avoid interpolation

---

## Troubleshooting

### Common Issues

#### ❌ "NetCDF file not found"
```
ERROR: NetCDF file not found: cmb_temp.nc
```
**✅ Solution**: Create sample files first:
```julia
include("examples/create_sample_netcdf_boundaries.jl")
```

#### ❌ "Grid size mismatch"
```
ERROR: Inner boundary nlat (32) != config nlat (64)
```
**✅ Solutions**:
1. Match grid sizes: `create_optimized_config(32, 32, nlat=32, nlon=64)`
2. Or rely on automatic interpolation (may be slower)

#### ❌ "SHTnsKit not found"
```
ERROR: Package SHTnsKit not found
```
**✅ Solution**: Install local dependency:
```julia
using Pkg
Pkg.develop(path="../SHTnsKit.jl")  # Adjust path as needed
```

#### ❌ Performance warnings
```
WARNING: Type instability detected
```
**✅ Solution**: Check field types:
```julia
temp_field = create_shtns_temperature_field(Float64, config, domain)  # Explicit Float64
```

### Debugging Tips

1. **Start Simple**: Begin with low resolution and uniform boundaries
2. **Check Boundaries**: Use `print_boundary_info()` to inspect loaded data
3. **Monitor Performance**: Use `@timed_transform` and `print_performance_report()`
4. **✅ Validate Data**: Use `get_boundary_statistics()` to check loaded values
5. **📝 Check Logs**: Look for warning messages during simulation setup

### Getting Help

- **Documentation**: Check the `docs/` directory for detailed guides
- **Examples**: Run `examples/` scripts to see working code
- **🐛 Issues**: Report bugs on the [GitHub repository](https://github.com/subhk/Geodynamo.jl)
- **💬 Community**: Ask questions in [GitHub Discussions](https://github.com/subhk/Geodynamo.jl/discussions)

---

## API Reference

### Main Configuration Functions

```julia
# Core setup
create_optimized_config(lmax, mmax; nlat, nlon, use_threading=true, use_simd=true)
create_radial_domain(inner_radius, outer_radius, nr)

# Field creation
create_shtns_temperature_field(precision, config, domain)
create_shtns_composition_field(precision, config, domain)
create_shtns_velocity_fields(precision, config, domain)
create_shtns_magnetic_fields(precision, config, domain)
```

### Boundary Condition Functions

```julia
# NetCDF boundaries
load_temperature_boundaries(inner_file, outer_file; precision=Float64)
load_composition_boundaries(inner_file, outer_file; precision=Float64)

# Hybrid boundaries (mix NetCDF + programmatic)
create_hybrid_temperature_boundaries(inner_spec, outer_spec, config; precision=Float64)
create_hybrid_composition_boundaries(inner_spec, outer_spec, config; precision=Float64)

# Programmatic boundaries
create_programmatic_boundary(pattern, config; amplitude, parameters=Dict())

# Single boundary loading
load_single_temperature_boundary(file_path, boundary_type; precision=Float64)
load_single_composition_boundary(file_path, boundary_type; precision=Float64)

# Time-dependent boundaries
create_time_dependent_programmatic_boundary(pattern, config, time_span, ntime; amplitude, parameters)
```

### Application Functions

```julia
# Apply boundaries to fields
apply_netcdf_temperature_boundaries!(temp_field, boundary_set, current_time=0.0)
apply_netcdf_composition_boundaries!(comp_field, boundary_set, current_time=0.0)

# Time evolution updates
update_temperature_boundaries_from_netcdf!(temp_field, boundary_set, timestep, dt)
update_composition_boundaries_from_netcdf!(comp_field, boundary_set, timestep, dt)
```

### Utility Functions

```julia
# Information and validation
print_boundary_info(boundary_set)
get_boundary_statistics(boundary_data)
validate_netcdf_temperature_compatibility(boundary_set, config)
validate_netcdf_composition_compatibility(boundary_set, config)

# Performance monitoring
reset_performance_stats!()
get_performance_summary()
print_performance_report()
@timed_transform expr
```

### Data Structures

```julia
# Main types
BoundaryData{T}                    # Single boundary data (inner or outer)
BoundaryConditionSet{T}            # Complete boundary set (inner + outer)
SHTnsConfig                        # SHTns configuration
SHTnsTemperatureField{T}           # Temperature field structure
SHTnsCompositionField{T}           # Composition field structure
```

---

## Advanced Topics

### Creating Custom NetCDF Files

For your own observational or model data:

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
    
    # Add metadata
    ds.attrib["title"] = "My Boundary Conditions"
    ds.attrib["description"] = "Custom boundary data"
    ds.attrib["created_by"] = "My Analysis"
end
```

### MPI Best Practices

```bash
# Run with MPI
mpirun -np 8 julia --project=. my_simulation.jl

# For SLURM clusters
srun -n 64 julia --project=. large_simulation.jl
```

```julia
# MPI-aware simulation setup
using MPI
MPI.Init()

if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    println("Starting geodynamo simulation on $(MPI.Comm_size(MPI.COMM_WORLD)) processes")
end

# Configuration automatically handles MPI decomposition
config = create_optimized_config(256, 256, nlat=512, nlon=1024)
```

---

## Learning Path

### Beginner (Learning the Basics)
1. **Start simple**: Use uniform boundaries and low resolution (lmax=16-32)
2. **Run examples**: Execute provided example scripts to understand workflow
3. **Explore patterns**: Try different programmatic boundary patterns
4. **Monitor performance**: Use built-in performance monitoring tools

### Intermediate (Adding Realism)
1. **Use NetCDF files**: Load boundary conditions from data files
2. **Mix approaches**: Combine NetCDF and programmatic boundaries
3. **Increase resolution**: Move to lmax=64-128 for more realistic simulations
4. **Add composition**: Include compositional convection alongside thermal

### Advanced (Production Simulations)
1. **Create custom patterns**: Develop your own boundary condition functions
2. **Time-dependent boundaries**: Use evolving boundary conditions
3. **High resolution**: Scale to lmax=256+ with MPI parallelization
4. **Optimize performance**: Fine-tune for your specific hardware

### Expert (Research & Development)
1. **Develop new physics**: Add magnetic field generation and dynamo action
2. **Custom analysis**: Create specialized post-processing and visualization tools
3. **Contribute**: Develop new features and optimizations for the community
4. **Publication**: Use for cutting-edge geophysical research

---

## Quick Reference

### Essential Command Cheat Sheet

```julia
# ===== CONFIGURATION =====
config = create_optimized_config(lmax, mmax, nlat=nlat, nlon=nlon)
domain = create_radial_domain(ri, ro, nr)

# ===== FIELD CREATION =====
temp_field = create_shtns_temperature_field(Float64, config, domain)
comp_field = create_shtns_composition_field(Float64, config, domain)

# ===== BOUNDARIES - PROGRAMMATIC =====
boundaries = create_hybrid_temperature_boundaries(
    (:pattern, amplitude, Dict("param"=>value)),   # Inner
    (:pattern, amplitude),                         # Outer
    config
)

# ===== BOUNDARIES - NETCDF =====
boundaries = load_temperature_boundaries("inner.nc", "outer.nc")

# ===== BOUNDARIES - HYBRID =====
boundaries = create_hybrid_temperature_boundaries(
    "data_file.nc",           # NetCDF file
    (:uniform, 300.0),        # Programmatic
    config
)

# ===== APPLICATION =====
apply_netcdf_temperature_boundaries!(temp_field, boundaries)

# ===== PERFORMANCE MONITORING =====
reset_performance_stats!()
@timed_transform begin
    # simulation code
end
print_performance_report()
```

### Pattern Reference

| Pattern | Code | Use Case |
|---------|------|----------|
| **Uniform** | `(:uniform, 300.0)` | Constant temperature |
| **Y₁₁** | `(:y11, 4000.0, Dict("amplitude"=>200))` | Simple convection |
| **Plume** | `(:plume, 4200.0, Dict("width"=>π/6))` | Mantle plumes |
| **Hemisphere** | `(:hemisphere, 1000.0, Dict("axis"=>"z"))` | Hemispherical structure |
| **Custom** | `(:custom, 100.0, Dict("function"=>func))` | Your own physics |

---

## Related Resources

- **[SHTnsKit.jl](https://github.com/subhk/SHTnsKit.jl)**: Spherical harmonic transform library
- **[Julia Language](https://julialang.org/)**: High-performance programming language
- **[NetCDF](https://www.unidata.ucar.edu/software/netcdf/)**: Scientific data format
- **[MPI.jl](https://github.com/JuliaParallel/MPI.jl)**: Message passing interface for Julia

---

## Citation

If you use Geodynamo.jl in your research, please cite:

```bibtex
@software{geodynamo_jl,
  title = {Geodynamo.jl: High-Performance Earth's Magnetic Field Simulation},
  author = {Kar, Subhajit},
  year = {2024},
  url = {https://github.com/subhk/Geodynamo.jl},
  note = {Julia package for geodynamo modeling}
}
```

---

## License

Geodynamo.jl is licensed under the [MIT License](LICENSE).

---

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on how to:

- Report bugs and request features
- Submit code improvements  
- Add documentation and examples
- Participate in discussions

---

**Happy geodynamo modeling!**

*For questions, issues, or collaboration opportunities, please visit our [GitHub repository](https://github.com/subhk/Geodynamo.jl).*