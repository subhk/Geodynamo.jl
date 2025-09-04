---
layout: page
title: Getting Started
permalink: /getting-started/
nav_order: 2
description: "Complete beginner's guide to geodynamo modeling with Geodynamo.jl"
---

# Getting Started Guide

Complete beginner's guide to geodynamo modeling with Geodynamo.jl.

---

## What You'll Learn

By the end of this guide, you'll be able to:
- Set up and run basic geodynamo simulations
- Apply different types of boundary conditions
- Visualize simulation results
- Understand key concepts in geodynamo modeling

## Prerequisites

- **Julia 1.8+** installed ([Download here](https://julialang.org/downloads/))
- Basic familiarity with Julia syntax
- Understanding of spherical coordinates (θ, φ, r)

---

## Installation

### Step 1: Install Geodynamo.jl

```julia
# Open Julia REPL and install the package
using Pkg
Pkg.add("Geodynamo")

# Install optional dependencies
Pkg.add("NCDatasets")  # For NetCDF support
Pkg.add("Plots")       # For visualization
```

### Step 2: Test Installation

```julia
using Geodynamo
println("Geodynamo.jl installed successfully!")
```

### Step 3: Install Visualization Tools (Optional)

```julia
Pkg.add(["PlotlyJS", "GR"])  # Additional plotting backends
```

---

## Core Concepts

### What is a Geodynamo?

The **geodynamo** is the process by which Earth's magnetic field is generated through convective motion in the liquid outer core. Key components include:

- **Thermal convection** - Hot material rises from the core-mantle boundary
- **Compositional convection** - Light elements are released during inner core crystallization  
- **Magnetic field generation** - Moving electrically conducting fluid generates magnetic fields
- **Coriolis effects** - Earth's rotation organizes the flow patterns

### Coordinate System

Geodynamo.jl uses **spherical coordinates**:
- **r** - Radial distance (normalized: inner core boundary = 0.35, surface = 1.0)
- **θ** - Colatitude (0 = North Pole, π = South Pole)  
- **φ** - Longitude (0 to 2π, eastward)

### Spectral Methods

The package uses **spherical harmonic transforms** to efficiently solve equations on the sphere:
- **Physical space** - Values at grid points (nlat × nlon × nr)
- **Spectral space** - Spherical harmonic coefficients (Y_l^m modes)
- **Transforms** - Convert between physical and spectral representations

---

## 🏗Basic Setup Workflow

Every Geodynamo.jl simulation follows this pattern:

```julia
# 1. Configuration
config = create_optimized_config(lmax, mmax, nlat=nlat, nlon=nlon)

# 2. Domain  
domain = create_radial_domain(inner_radius, outer_radius, nr)

# 3. Fields
temp_field = create_shtns_temperature_field(Float64, config, domain)

# 4. Boundary conditions
boundaries = create_hybrid_temperature_boundaries(inner_spec, outer_spec, config)

# 5. Apply boundaries
apply_netcdf_temperature_boundaries!(temp_field, boundaries)

# 6. Run simulation
# ... physics calculations ...
```

---

## 🎮 Your First Simulation

Let's build a complete basic simulation step by step.

### Step 1: Load the Package

```julia
using Geodynamo
```

### Step 2: Create Configuration

The configuration defines the resolution and numerical parameters:

```julia
# Resolution parameters
lmax = 32       # Maximum spherical harmonic degree
mmax = 32       # Maximum spherical harmonic order  
nlat = 64       # Number of latitude points
nlon = 128      # Number of longitude points

# Create optimized configuration
config = create_optimized_config(lmax, mmax, 
                                nlat=nlat, nlon=nlon,
                                use_threading=true,
                                use_simd=true)

println("Configuration created:")
println("   Spectral resolution: lmax=$lmax, mmax=$mmax") 
println("   Physical grid: $(nlat)×$(nlon)")
println("   Total spectral modes: $(config.nlm)")
```

**Resolution Guidelines:**
- **Learning**: lmax=16-32, nlat=32-64 (fast, ~100 MB memory)
- **Research**: lmax=64-128, nlat=128-256 (realistic, ~1-4 GB memory) 
- **Production**: lmax=256+, nlat=512+ (high-resolution, ~16+ GB memory)

### Step 3: Define Domain

The radial domain represents the spherical shell where convection occurs:

```julia
# Earth-like parameters
inner_radius = 0.35    # Inner core boundary (normalized)
outer_radius = 1.0     # Core-mantle boundary (normalized)  
nr = 64                # Number of radial grid points

domain = create_radial_domain(inner_radius, outer_radius, nr)

println("Domain created:")
println("   Shell: r ∈ [$inner_radius, $outer_radius]")
println("   Radial points: $nr")
println("   Shell thickness: $(outer_radius - inner_radius) (normalized)")
```

### Step 4: Create Temperature Field

```julia
# Create temperature field for thermal convection
temp_field = create_shtns_temperature_field(Float64, config, domain)

println("Temperature field created")
```

### Step 5: Set Boundary Conditions

Start with simple uniform boundary conditions:

```julia
# Simple thermal boundary conditions
cmb_temperature = 4000.0      # Hot core-mantle boundary (K)
surface_temperature = 300.0   # Cool surface (K)

temp_boundaries = create_hybrid_temperature_boundaries(
    (:uniform, cmb_temperature),     # Inner boundary: uniform hot
    (:uniform, surface_temperature), # Outer boundary: uniform cool
    config
)

println("Boundary conditions created:")
println("   CMB temperature: $cmb_temperature K")
println("   Surface temperature: $surface_temperature K")
```

### Step 6: Apply Boundaries

```julia
# Apply boundary conditions to the field
apply_netcdf_temperature_boundaries!(temp_field, temp_boundaries)

# Verify application
print_boundary_info(temp_boundaries)

println("Boundary conditions applied successfully!")
```

### Step 7: Complete First Simulation

```julia
function my_first_geodynamo_simulation()
    println("My First Geodynamo Simulation")
    println("=" ^ 40)
    
    # Configuration
    config = create_optimized_config(32, 32, nlat=64, nlon=128)
    domain = create_radial_domain(0.35, 1.0, 64)
    
    # Create temperature field  
    temp_field = create_shtns_temperature_field(Float64, config, domain)
    
    # Simple boundary conditions
    temp_boundaries = create_hybrid_temperature_boundaries(
        (:uniform, 4000.0),  # Hot CMB
        (:uniform, 300.0),   # Cool surface
        config
    )
    
    # Apply boundaries
    apply_netcdf_temperature_boundaries!(temp_field, temp_boundaries)
    
    # Summary
    stats = get_boundary_statistics(temp_boundaries.inner_boundary)
    println("Simulation Summary:")
    println("   Resolution: lmax=$(config.lmax), nlat=$(config.nlat)")
    println("   Temperature range: [$(round(stats["min"])), $(round(stats["max"]))] K")
    println("   Memory usage: ~$((config.nlm * 64 * 8) ÷ 1024 ÷ 1024) MB per field")
    
    println("\nFirst simulation setup complete!")
    
    return temp_field, temp_boundaries, config, domain
end

# Run your first simulation
temp_field, boundaries, config, domain = my_first_geodynamo_simulation()
```

**Expected Output:**
```
My First Geodynamo Simulation  
========================================
Simulation Summary:
   Resolution: lmax=32, nlat=64
   Temperature range: [4000, 4000] K
   Memory usage: ~8 MB per field

First simulation setup complete!
```

---

## Adding Realistic Physics

### Realistic Boundary Conditions

Instead of uniform boundaries, let's add realistic structure:

```julia
function realistic_boundary_example()
    println("Realistic Boundary Conditions Example")
    
    config = create_optimized_config(64, 64, nlat=128, nlon=256)
    domain = create_radial_domain(0.35, 1.0, 64)
    
    # Hot plume at core-mantle boundary
    temp_boundaries = create_hybrid_temperature_boundaries(
        # Inner: Hot plume pattern
        (:plume, 4200.0, Dict(
            "center_theta" => π/3,      # 60° from north pole
            "center_phi" => 0.0,        # 0° longitude (Greenwich)
            "width" => π/6,             # 30° plume width  
            "amplitude" => 300.0        # 300 K temperature excess
        )),
        # Outer: Uniform cool surface
        (:uniform, 300.0),
        config
    )
    
    temp_field = create_shtns_temperature_field(Float64, config, domain)
    apply_netcdf_temperature_boundaries!(temp_field, temp_boundaries)
    
    # Analysis
    inner_stats = get_boundary_statistics(temp_boundaries.inner_boundary)
    
    println("Plume Boundary Analysis:")
    println("   Base CMB temperature: 4200 K")
    println("   Plume temperature range: [$(round(inner_stats["min"])), $(round(inner_stats["max"]))] K")
    println("   Temperature contrast: $(round(inner_stats["max"] - inner_stats["min"])) K")
    println("   Standard deviation: $(round(inner_stats["std"])) K")
    
    print_boundary_info(temp_boundaries)
    
    return temp_field, temp_boundaries
end

# Run realistic example
temp_field, boundaries = realistic_boundary_example()
```

### Available Boundary Patterns

| Pattern | Description | Parameters |
|---------|-------------|------------|
| `:uniform` | Constant temperature | `amplitude` |
| `:y11` | Y₁₁ spherical harmonic | `amplitude` (perturbation) |
| `:plume` | Gaussian hotspot | `width`, `center_theta`, `center_phi` |
| `:hemisphere` | North/south contrast | `axis` ("x", "y", or "z") |
| `:dipole` | Dipolar pattern (Y₁₀) | `amplitude` |
| `:custom` | Your own function | `function` |

**Example with multiple patterns:**
```julia
# Y11 spherical harmonic pattern  
y11_boundaries = create_hybrid_temperature_boundaries(
    (:y11, 4000.0, Dict("amplitude" => 200.0)),  # Y₁₁ with 200K amplitude
    (:uniform, 300.0),
    config
)

# Hemispherical contrast
hemisphere_boundaries = create_hybrid_temperature_boundaries(
    (:hemisphere, 4000.0, Dict(
        "axis" => "z",         # North-south contrast
        "amplitude" => 400.0   # Temperature difference
    )),
    (:uniform, 300.0), 
    config
)
```

---

## Working with Real Data

### Using NetCDF Files

For realistic simulations, you can load boundary conditions from NetCDF files:

```julia
function netcdf_boundary_example()
    println("NetCDF Boundary Conditions Example")
    
    # Check if sample files exist  
    cmb_file = "cmb_temp.nc"
    surface_file = "surface_temp.nc"
    
    if !isfile(cmb_file) || !isfile(surface_file)
        println("📥 Creating sample NetCDF files...")
        # Create sample files (run this once)
        include("examples/create_sample_netcdf_boundaries.jl")
        println("Sample files created")
    end
    
    config = create_optimized_config(64, 64, nlat=128, nlon=256)
    domain = create_radial_domain(0.35, 1.0, 64)
    
    # Load boundaries from NetCDF files
    println("Loading temperature boundaries...")
    temp_boundaries = load_temperature_boundaries(cmb_file, surface_file)
    
    # Validate compatibility  
    is_compatible = validate_netcdf_temperature_compatibility(temp_boundaries, config)
    println("Boundary compatibility: $(is_compatible ? "" : "")")
    
    if !is_compatible
        println("⚠Grid size mismatch - automatic interpolation will be used")
    end
    
    # Apply boundaries
    temp_field = create_shtns_temperature_field(Float64, config, domain)
    apply_netcdf_temperature_boundaries!(temp_field, temp_boundaries)
    
    # Detailed analysis
    println("\nNetCDF Boundary Analysis:")
    print_boundary_info(temp_boundaries)
    
    return temp_field, temp_boundaries
end

# Run NetCDF example (requires sample files)
# temp_field, boundaries = netcdf_boundary_example()
```

### Hybrid Approaches

Mix NetCDF data with programmatic patterns for maximum flexibility:

```julia
function hybrid_boundary_example()
    println("Hybrid Boundary Conditions Example")
    
    config = create_optimized_config(64, 64, nlat=128, nlon=256)
    domain = create_radial_domain(0.35, 1.0, 64)
    
    # Strategy 1: NetCDF inner + programmatic outer
    if isfile("cmb_temp.nc")
        temp_boundaries_1 = create_hybrid_temperature_boundaries(
            "cmb_temp.nc",        # Complex CMB from data
            (:uniform, 300.0),    # Simple surface
            config
        )
        println("Strategy 1: NetCDF inner + programmatic outer")
    end
    
    # Strategy 2: Programmatic inner + NetCDF outer
    temp_boundaries_2 = create_hybrid_temperature_boundaries(
        (:plume, 4200.0, Dict("width" => π/6, "center_theta" => π/3)),
        (:uniform, 300.0),    # Would be NetCDF if file exists
        config  
    )
    println("Strategy 2: Programmatic inner + simple outer")
    
    # Apply the available strategy
    temp_field = create_shtns_temperature_field(Float64, config, domain)
    apply_netcdf_temperature_boundaries!(temp_field, temp_boundaries_2)
    
    return temp_field, temp_boundaries_2
end

# Run hybrid example
temp_field, boundaries = hybrid_boundary_example()
```

---

## Running Simulations

### Basic Time-Stepping Loop

Here's how to evolve your simulation in time:

```julia
function basic_simulation_loop()
    println("Basic Simulation Time Loop")
    
    # Setup (reuse from previous examples)
    config = create_optimized_config(32, 32, nlat=64, nlon=128)
    domain = create_radial_domain(0.35, 1.0, 64)
    
    # Create fields
    temp_field = create_shtns_temperature_field(Float64, config, domain)
    vel_field = create_shtns_velocity_fields(Float64, config, domain)
    
    # Boundary conditions
    temp_boundaries = create_hybrid_temperature_boundaries(
        (:plume, 4200.0, Dict("width" => π/8, "center_theta" => π/3)),
        (:uniform, 300.0),
        config
    )
    apply_netcdf_temperature_boundaries!(temp_field, temp_boundaries)
    
    # Simulation parameters
    dt = 0.001           # Time step
    nsteps = 100         # Number of steps
    output_freq = 20     # Output every N steps
    
    println("Starting simulation loop...")
    println("   Time step: $dt")
    println("   Total steps: $nsteps") 
    println("   Output frequency: every $output_freq steps")
    
    # Time loop
    for step in 1:nsteps
        current_time = step * dt
        
        # Physics calculations (simplified)
        if step > 1  # Skip first step
            # Compute nonlinear terms
            compute_temperature_nonlinear!(temp_field, vel_field)
            
            # Time stepping would happen here
            # solve_implicit_step!(temp_field, dt)
        end
        
        # Output and monitoring
        if step % output_freq == 0
            println("   Step $step/$nsteps: t=$(round(current_time, digits=3))")
            
            # Compute diagnostics
            inner_stats = get_boundary_statistics(temp_boundaries.inner_boundary)
            println("     CMB T_max = $(round(inner_stats["max"])) K")
        end
    end
    
    println("Simulation loop completed!")
    return temp_field, vel_field
end

# Run simulation loop
temp_field, vel_field = basic_simulation_loop()
```

### Performance Monitoring

Monitor your simulation performance:

```julia
function monitored_simulation()
    println("Performance-Monitored Simulation")
    
    # Setup
    config = create_optimized_config(64, 64, nlat=128, nlon=256)
    domain = create_radial_domain(0.35, 1.0, 64)
    
    temp_field = create_shtns_temperature_field(Float64, config, domain)
    vel_field = create_shtns_velocity_fields(Float64, config, domain)
    
    boundaries = create_hybrid_temperature_boundaries(
        (:uniform, 4000.0), (:uniform, 300.0), config
    )
    apply_netcdf_temperature_boundaries!(temp_field, boundaries)
    
    # Initialize performance monitoring
    reset_performance_stats!()
    
    println("Running monitored simulation...")
    
    # Timed simulation
    elapsed = @elapsed @timed_transform begin
        for step in 1:50  # Shorter run for monitoring
            compute_temperature_nonlinear!(temp_field, vel_field)
        end
    end
    
    # Performance report
    println("\nPerformance Report:")
    print_performance_report()
    
    println("\nPerformance Summary:")
    println("   Total time: $(round(elapsed, digits=2)) seconds")
    println("   Average per step: $(round(elapsed/50*1000, digits=2)) ms")
    
    return temp_field
end

# Run monitored simulation  
temp_field = monitored_simulation()
```

---

## Visualization and Analysis

### Basic Plotting

Let's create some basic visualizations:

```julia
using Plots
plotlyjs()  # Use PlotlyJS backend

function basic_visualization_example()
    println("Basic Visualization Example")
    
    # Create a simple temperature field for plotting
    config = create_optimized_config(32, 32, nlat=64, nlon=128)
    domain = create_radial_domain(0.35, 1.0, 32)
    
    temp_field = create_shtns_temperature_field(Float64, config, domain)
    
    # Apply plume boundary  
    boundaries = create_hybrid_temperature_boundaries(
        (:plume, 4200.0, Dict(
            "width" => π/6,
            "center_theta" => π/3,
            "center_phi" => π/4
        )),
        (:uniform, 300.0),
        config
    )
    apply_netcdf_temperature_boundaries!(temp_field, boundaries)
    
    # Get boundary data for plotting
    inner_data = boundaries.inner_boundary.data
    theta = boundaries.inner_boundary.theta
    phi = boundaries.inner_boundary.phi
    
    # Convert to degrees for plotting
    theta_deg = rad2deg.(theta)
    phi_deg = rad2deg.(phi .- π)  # Center at 0° longitude
    
    # Create heatmap
    p1 = heatmap(phi_deg, theta_deg, inner_data',
                title="CMB Temperature (Plume Pattern)",
                xlabel="Longitude (°)",
                ylabel="Colatitude (°)",
                c=:plasma,
                aspect_ratio=:equal)
    
    # Radial profile (if we had 3D data)
    r_points = domain.r_vector
    temp_profile = 4000 .- 3700 * (r_points .- 0.35) / (1.0 - 0.35)  # Linear profile
    
    p2 = plot(r_points, temp_profile,
             title="Radial Temperature Profile",
             xlabel="Radius (normalized)",
             ylabel="Temperature (K)",
             linewidth=2, marker=:circle)
    
    # Combine plots
    combined = plot(p1, p2, layout=(1,2), size=(800,400))
    
    # Save plot
    savefig(combined, "basic_geodynamo_plot.png")
    println("Plot saved as: basic_geodynamo_plot.png")
    
    return combined
end

# Create visualization
plot_result = basic_visualization_example()
```

### Using Built-in Plotting Scripts

Geodynamo.jl includes specialized plotting scripts:

```bash
# Plot spherical surface at constant radius
julia --project=. script/plot_sphere_r.jl output.nc --quantity=temperature --r=1.0 --out=surface_temp.png

# Create Hammer projection of magnetic field  
julia --project=. script/plot_hammer_magnetic.jl output.nc --r=1.0 --out=magnetic_hammer.png

# Plot slice at constant z
julia --project=. script/plot_slice_z.jl output.nc --quantity=temperature --z=0.5 --out=temp_slice.png
```

---

## Learning Exercises

### Exercise 1: Resolution Study

Compare different resolutions:

```julia
function resolution_study()
    println("Exercise 1: Resolution Study")
    
    resolutions = [
        (16, 32, 64),    # lmax, nlat, nlon
        (32, 64, 128),
        (64, 128, 256)
    ]
    
    for (lmax, nlat, nlon) in resolutions
        config = create_optimized_config(lmax, lmax, nlat=nlat, nlon=nlon)
        domain = create_radial_domain(0.35, 1.0, 32)
        
        temp_field = create_shtns_temperature_field(Float64, config, domain)
        
        # Memory estimate
        memory_mb = (config.nlm * 32 * 8) ÷ 1024 ÷ 1024
        
        println("   lmax=$lmax, grid=$(nlat)×$(nlon): $(config.nlm) modes, ~$(memory_mb) MB")
    end
    
    println("Key insight: Memory scales as lmax² while accuracy improves")
end

resolution_study()
```

### Exercise 2: Boundary Condition Comparison

Compare different boundary patterns:

```julia
function boundary_comparison_study()
    println("Exercise 2: Boundary Pattern Comparison")
    
    config = create_optimized_config(32, 32, nlat=64, nlon=128)
    domain = create_radial_domain(0.35, 1.0, 32)
    
    patterns = [
        ("Uniform", (:uniform, 4000.0)),
        ("Y11", (:y11, 4000.0, Dict("amplitude" => 200.0))),  
        ("Plume", (:plume, 4200.0, Dict("width" => π/6, "center_theta" => π/3))),
        ("Hemisphere", (:hemisphere, 4000.0, Dict("axis" => "z", "amplitude" => 300.0)))
    ]
    
    println("Pattern Comparison:")
    for (name, pattern_spec) in patterns
        boundaries = create_hybrid_temperature_boundaries(
            pattern_spec, (:uniform, 300.0), config
        )
        
        stats = get_boundary_statistics(boundaries.inner_boundary)
        
        println("   $name:")
        println("     Range: [$(round(stats["min"])), $(round(stats["max"]))] K")
        println("     Std Dev: $(round(stats["std"], digits=1)) K")
    end
    
    println("Key insight: Different patterns create different convection styles")
end

boundary_comparison_study()
```

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: "Package not found"
```julia
# Error: Package Geodynamo not found
```
**Solution:**
```julia
using Pkg
Pkg.add("Geodynamo")  # Install package first
```

#### Issue 2: "Grid size mismatch"  
```julia
# Error: Inner boundary nlat (32) != config nlat (64)
```
**Solutions:**
```julia
# Option 1: Match grid sizes
config = create_optimized_config(32, 32, nlat=32, nlon=64)

# Option 2: Let the package interpolate (slower)
# The package will automatically interpolate mismatched grids
```

#### Issue 3: "Memory allocation error"
```julia
# Error: Out of memory
```
**Solutions:**
```julia
# Reduce resolution
config = create_optimized_config(16, 16, nlat=32, nlon=64)  # Smaller

# Use Float32 instead of Float64
temp_field = create_shtns_temperature_field(Float32, config, domain)
```

#### Issue 4: "Performance too slow"
**Solutions:**
```julia
# Enable all optimizations
config = create_optimized_config(32, 32, nlat=64, nlon=128,
                                use_threading=true,
                                use_simd=true)

# Check threading
println("Julia threads: $(Threads.nthreads())")
# Start Julia with: julia -t 8 (for 8 threads)
```

### Getting Help

1. **Check documentation**: All functions have built-in help
   ```julia
   help?> create_optimized_config
   ```

2. **Use validation functions**:
   ```julia
   validate_netcdf_temperature_compatibility(boundaries, config)
   ```

3. **Print debug information**:
   ```julia
   print_boundary_info(boundaries)
   get_boundary_statistics(boundaries.inner_boundary)
   ```

4. **Community support**:
   - [GitHub Issues](https://github.com/subhk/Geodynamo.jl/issues)
   - [Julia Discourse](https://discourse.julialang.org/)

---

## Next Steps

### Beginner Path
1. Complete this getting started guide
2. Try the [visualization examples](visualization.html)
3. Explore [basic examples](examples.html)
4. 📖 Read the [API reference](api-reference.html)

### Intermediate Path  
1. Use NetCDF boundary conditions
2. ⏰ Add time-dependent boundaries
3. Experiment with hybrid approaches
4. Perform resolution studies

### Advanced Path
1. Add magnetic field generation
2. Optimize for HPC clusters
3. Develop new physics modules
4. Contribute to the project

---

## Congratulations!

You've completed the Geodynamo.jl getting started guide! You now know how to:

- Set up basic geodynamo simulations
- Apply different boundary conditions  
- Run time-stepping loops
- Visualize results
- Monitor performance
- Troubleshoot common issues

**Ready for more?** Check out our [Examples Gallery](examples.html) for working code samples, or dive into the [API Reference](api-reference.html) for detailed documentation.

---

*Happy geodynamo modeling! *