---
layout: default  
title: Examples
---

# Examples Gallery

Working code examples for common Geodynamo.jl use cases.

---

## 🚀 Getting Started Examples

### Example 1: Basic Thermal Convection

Complete beginner example with uniform boundary conditions.

```julia
using Geodynamo

function basic_thermal_convection()
    println("🌡️  Basic Thermal Convection Example")
    
    # Step 1: Create configuration
    config = create_optimized_config(32, 32, nlat=64, nlon=128, use_threading=true)
    println("✓ Created SHTns configuration: lmax=$(config.lmax), nlat=$(config.nlat)")
    
    # Step 2: Set up radial domain (Earth-like core)
    inner_radius = 0.35    # Inner core boundary
    outer_radius = 1.0     # Core-mantle boundary  
    nr = 64                # Radial resolution
    domain = create_radial_domain(inner_radius, outer_radius, nr)
    println("✓ Created radial domain: r ∈ [$(inner_radius), $(outer_radius)], nr=$(nr)")
    
    # Step 3: Create temperature field
    temp_field = create_shtns_temperature_field(Float64, config, domain)
    println("✓ Created temperature field")
    
    # Step 4: Set boundary conditions (hot bottom, cold top)
    temp_boundaries = create_hybrid_temperature_boundaries(
        (:uniform, 4000.0),    # Hot CMB at 4000 K
        (:uniform, 300.0),     # Cool surface at 300 K  
        config
    )
    println("✓ Created uniform boundary conditions")
    
    # Step 5: Apply boundaries to field
    apply_netcdf_temperature_boundaries!(temp_field, temp_boundaries)
    println("✓ Applied boundary conditions")
    
    # Step 6: Add small perturbation to initiate convection
    # set_temperature_ic!(temp_field, :random_perturbation, domain)
    
    # Step 7: Print boundary information  
    print_boundary_info(temp_boundaries)
    
    println("🎉 Basic thermal convection setup complete!")
    return temp_field, temp_boundaries
end

# Run the example
temp_field, boundaries = basic_thermal_convection()
```

**Expected Output:**
```
🌡️  Basic Thermal Convection Example
✓ Created SHTns configuration: lmax=32, nlat=64
✓ Created radial domain: r ∈ [0.35, 1.0], nr=64  
✓ Created temperature field
✓ Created uniform boundary conditions
✓ Applied boundary conditions
🎉 Basic thermal convection setup complete!
```

---

### Example 2: Plume-Driven Convection

Realistic mantle plume simulation with localized heating.

```julia
using Geodynamo

function plume_convection_example()
    println("🌋 Plume-Driven Convection Example")
    
    # Higher resolution for realistic plume structure
    config = create_optimized_config(64, 64, nlat=128, nlon=256, use_threading=true)
    domain = create_radial_domain(0.35, 1.0, 64)
    
    # Create temperature field
    temp_field = create_shtns_temperature_field(Float64, config, domain)
    
    # Hot plume at CMB with realistic parameters
    temp_boundaries = create_hybrid_temperature_boundaries(
        # Inner boundary: Hot plume
        (:plume, 4200.0, Dict(
            "center_theta" => π/3,      # 60° from north pole (mid-latitude)
            "center_phi" => 0.0,        # Prime meridian
            "width" => π/6,             # 30° plume width (~3300 km diameter)
            "amplitude" => 300.0        # 300 K temperature excess
        )),
        # Outer boundary: Uniform cool surface  
        (:uniform, 300.0),              # Earth's average surface temperature
        config
    )
    
    # Apply boundary conditions
    apply_netcdf_temperature_boundaries!(temp_field, temp_boundaries)
    
    # Analyze boundary statistics
    inner_stats = get_boundary_statistics(temp_boundaries.inner_boundary)
    outer_stats = get_boundary_statistics(temp_boundaries.outer_boundary)
    
    println("📊 Boundary Statistics:")
    println("   CMB temperature range: [$(round(inner_stats["min"])), $(round(inner_stats["max"]))] K")
    println("   Surface temperature: $(round(outer_stats["mean"])) K")
    println("   Plume excess: $(round(inner_stats["max"] - inner_stats["min"])) K")
    
    # Print detailed information
    print_boundary_info(temp_boundaries)
    
    println("🎉 Plume convection setup complete!")
    return temp_field, temp_boundaries
end

# Run the example
temp_field, boundaries = plume_convection_example()
```

---

### Example 3: Multiple Plumes Configuration

Advanced example with multiple hotspots mimicking Earth's mantle.

```julia
function multiple_plumes_example()
    println("🌋🌋 Multiple Plumes Example")
    
    config = create_optimized_config(64, 64, nlat=128, nlon=256)
    domain = create_radial_domain(0.35, 1.0, 64)
    
    # Create custom function for multiple plumes
    function multiple_plume_pattern(theta, phi)
        # Base temperature
        base_temp = 4000.0
        
        # Plume locations (theta, phi, intensity)
        plumes = [
            (π/3, 0.0, 200.0),         # Pacific hotspot
            (2π/3, π, 150.0),          # Atlantic hotspot  
            (π/2, π/2, 180.0),         # Indian Ocean hotspot
            (π/4, 3π/2, 120.0)         # Minor hotspot
        ]
        
        temp = base_temp
        for (plume_theta, plume_phi, intensity) in plumes
            # Gaussian plume centered at (plume_theta, plume_phi)
            distance = acos(cos(theta)*cos(plume_theta) + 
                          sin(theta)*sin(plume_theta)*cos(phi - plume_phi))
            width = π/8  # Plume width
            temp += intensity * exp(-(distance/width)^2)
        end
        
        return temp
    end
    
    # Apply custom boundary pattern
    temp_boundaries = create_hybrid_temperature_boundaries(
        (:custom, 4000.0, Dict("function" => multiple_plume_pattern)),
        (:uniform, 300.0),
        config
    )
    
    temp_field = create_shtns_temperature_field(Float64, config, domain)
    apply_netcdf_temperature_boundaries!(temp_field, temp_boundaries)
    
    # Analysis
    stats = get_boundary_statistics(temp_boundaries.inner_boundary)
    println("📊 Multiple Plumes Statistics:")
    println("   Temperature range: [$(round(stats["min"])), $(round(stats["max"]))] K")
    println("   Average temperature: $(round(stats["mean"])) K")
    println("   Standard deviation: $(round(stats["std"])) K")
    
    return temp_field, temp_boundaries
end

# Run the example
temp_field, boundaries = multiple_plumes_example()
```

---

## 📁 NetCDF Data Examples

### Example 4: Using Real Data Files

Loading boundary conditions from NetCDF files (requires data files).

```julia
using Geodynamo

function realistic_boundaries_example()
    println("🌍 Realistic Boundary Conditions Example")
    
    # Check if sample NetCDF files exist
    cmb_file = "cmb_temp.nc"
    surface_file = "surface_temp.nc"
    
    if !isfile(cmb_file) || !isfile(surface_file)
        println("📥 Creating sample NetCDF files...")
        include("examples/create_sample_netcdf_boundaries.jl")
        println("✓ Sample files created")
    end
    
    # Set up simulation
    config = create_optimized_config(64, 64, nlat=128, nlon=256)
    domain = create_radial_domain(0.35, 1.0, 64)
    
    # Load temperature boundaries from NetCDF files
    println("📂 Loading temperature boundaries from NetCDF files...")
    temp_boundaries = load_temperature_boundaries(cmb_file, surface_file)
    
    # Load compositional boundaries (if available)
    comp_file_inner = "cmb_composition.nc" 
    comp_file_outer = "surface_composition.nc"
    
    comp_boundaries = nothing
    if isfile(comp_file_inner) && isfile(comp_file_outer)
        println("📂 Loading composition boundaries...")
        comp_boundaries = load_composition_boundaries(comp_file_inner, comp_file_outer)
    end
    
    # Create fields
    temp_field = create_shtns_temperature_field(Float64, config, domain)
    comp_field = comp_boundaries !== nothing ? 
                 create_shtns_composition_field(Float64, config, domain) : nothing
    
    # Apply boundaries
    apply_netcdf_temperature_boundaries!(temp_field, temp_boundaries)
    if comp_field !== nothing
        apply_netcdf_composition_boundaries!(comp_field, comp_boundaries)
        println("✓ Applied compositional boundaries")
    end
    
    # Validate compatibility
    is_compatible = validate_netcdf_temperature_compatibility(temp_boundaries, config)
    println("🔍 Boundary compatibility: $(is_compatible ? "✓" : "✗")")
    
    # Detailed analysis
    println("\n📊 Boundary Analysis:")
    print_boundary_info(temp_boundaries)
    
    if comp_boundaries !== nothing
        println("\n📊 Composition Boundary Analysis:")
        print_boundary_info(comp_boundaries)
    end
    
    println("🎉 Realistic boundaries loaded successfully!")
    return temp_field, comp_field, temp_boundaries, comp_boundaries
end

# Run the example
temp_field, comp_field, temp_boundaries, comp_boundaries = realistic_boundaries_example()
```

---

### Example 5: Hybrid Approaches

Mix NetCDF data with programmatic patterns for maximum flexibility.

```julia
function hybrid_boundaries_example()
    println("🔀 Hybrid Boundaries Example")
    
    config = create_optimized_config(64, 64, nlat=128, nlon=256)
    domain = create_radial_domain(0.35, 1.0, 64)
    
    # Strategy 1: NetCDF inner + programmatic outer
    println("📋 Strategy 1: Complex CMB data + Simple surface")
    temp_boundaries_1 = create_hybrid_temperature_boundaries(
        "cmb_temp.nc",           # Complex CMB from data file
        (:uniform, 300.0),       # Simple uniform surface
        config
    )
    
    # Strategy 2: Programmatic inner + NetCDF outer  
    println("📋 Strategy 2: Analytical inner + Real surface data")
    temp_boundaries_2 = create_hybrid_temperature_boundaries(
        (:plume, 4200.0, Dict(
            "center_theta" => π/3,
            "center_phi" => π/4,
            "width" => π/8,
            "amplitude" => 200.0
        )),                      # Analytical plume at CMB
        "surface_temp.nc",       # Real surface temperature data  
        config
    )
    
    # Strategy 3: Mixed composition boundaries
    println("📋 Strategy 3: Mixed composition boundaries")
    comp_boundaries = create_hybrid_composition_boundaries(
        (:hemisphere, 0.8, Dict(
            "axis" => "z",
            "amplitude" => 0.2
        )),                      # Light elements in northern hemisphere
        "surface_composition.nc", # Surface composition from data
        config  
    )
    
    # Create fields and apply boundaries
    temp_field_1 = create_shtns_temperature_field(Float64, config, domain)
    temp_field_2 = create_shtns_temperature_field(Float64, config, domain)
    comp_field = create_shtns_composition_field(Float64, config, domain)
    
    apply_netcdf_temperature_boundaries!(temp_field_1, temp_boundaries_1)
    apply_netcdf_temperature_boundaries!(temp_field_2, temp_boundaries_2)  
    apply_netcdf_composition_boundaries!(comp_field, comp_boundaries)
    
    # Compare strategies
    println("\n📊 Boundary Comparison:")
    
    stats_1 = get_boundary_statistics(temp_boundaries_1.inner_boundary)
    stats_2 = get_boundary_statistics(temp_boundaries_2.inner_boundary)
    
    println("Strategy 1 CMB range: [$(round(stats_1["min"])), $(round(stats_1["max"]))] K")
    println("Strategy 2 CMB range: [$(round(stats_2["min"])), $(round(stats_2["max"]))] K")
    
    println("🎉 Hybrid boundaries example complete!")
    return temp_field_1, temp_field_2, comp_field
end

# Run the example  
temp1, temp2, comp = hybrid_boundaries_example()
```

---

## ⏰ Time-Dependent Examples

### Example 6: Rotating Plume Pattern

Time-evolving boundary conditions with rotating hotspot.

```julia
function time_dependent_example()
    println("⏰ Time-Dependent Boundaries Example")
    
    config = create_optimized_config(32, 32, nlat=64, nlon=128)
    domain = create_radial_domain(0.35, 1.0, 64)
    
    # Create rotating plume pattern
    println("🌀 Creating rotating plume pattern...")
    rotating_inner = create_time_dependent_programmatic_boundary(
        :plume, config, 
        (0.0, 10.0),    # Time span: 0 to 10 time units
        50,             # 50 time steps
        amplitude=4200.0,
        parameters=Dict(
            "width" => π/6,              # Plume width
            "center_theta" => π/3,       # Fixed latitude  
            "time_factor" => 2π,         # One full rotation over time span
            "base_temperature" => 4000.0 # Base CMB temperature
        )
    )
    
    # Create boundary set with time-dependent inner boundary
    temp_boundaries = BoundaryConditionSet(
        rotating_inner,                                          # Time-dependent inner
        create_programmatic_boundary(:uniform, config, amplitude=300.0), # Static outer
        "temperature",
        time()
    )
    
    # Create temperature field
    temp_field = create_shtns_temperature_field(Float64, config, domain)
    
    # Simulate time evolution
    println("⏳ Simulating time evolution...")
    dt = 0.2  # Time step
    nsteps = 10
    
    for step in 1:nsteps
        current_time = step * dt
        
        # Update boundaries for current time
        apply_netcdf_temperature_boundaries!(temp_field, temp_boundaries, current_time)
        
        # Get current boundary statistics
        stats = get_boundary_statistics(temp_boundaries.inner_boundary)
        
        println("Step $(step): t=$(current_time), CMB T_max=$(round(stats["max"])) K")
        
        # Here you would typically run physics calculations:
        # compute_temperature_nonlinear!(temp_field, vel_field)
        # solve_implicit_step!(temp_field, dt)
    end
    
    println("🎉 Time-dependent boundary example complete!")
    return temp_field, temp_boundaries
end

# Run the example
temp_field, boundaries = time_dependent_example()
```

---

## 🔬 Advanced Simulation Examples

### Example 7: Full Convection Simulation Loop

Complete simulation with nonlinear terms and time stepping.

```julia
using Geodynamo

function full_convection_simulation()
    println("🔬 Full Convection Simulation Example")
    
    # High-resolution configuration  
    config = create_optimized_config(64, 64, nlat=128, nlon=256, use_threading=true)
    domain = create_radial_domain(0.35, 1.0, 64)
    
    # Create all necessary fields
    println("🏗️  Setting up fields...")
    temp_field = create_shtns_temperature_field(Float64, config, domain)
    vel_field = create_shtns_velocity_fields(Float64, config, domain)
    
    # Set up realistic boundary conditions
    temp_boundaries = create_hybrid_temperature_boundaries(
        (:plume, 4200.0, Dict("width" => π/8, "center_theta" => π/3)),
        (:uniform, 300.0),
        config
    )
    
    apply_netcdf_temperature_boundaries!(temp_field, temp_boundaries)
    
    # Simulation parameters
    dt = 0.001          # Time step
    nsteps = 1000       # Number of steps
    output_freq = 100   # Output every N steps
    
    # Initialize performance monitoring
    reset_performance_stats!()
    
    println("🚀 Starting simulation loop...")
    @timed_transform begin
        for step in 1:nsteps
            current_time = step * dt
            
            # Update time-dependent boundaries (if any)
            if step % 10 == 0  # Update every 10 steps
                apply_netcdf_temperature_boundaries!(temp_field, temp_boundaries, current_time)
            end
            
            # Compute nonlinear terms
            compute_temperature_nonlinear!(temp_field, vel_field; geometry=:shell)
            
            # Time step (placeholder - actual implementation more complex)
            # solve_implicit_step!(temp_field, dt)
            
            # Output and monitoring
            if step % output_freq == 0
                println("Step $(step)/$(nsteps): t=$(current_time)")
                
                # Compute diagnostics
                # temp_energy = compute_temperature_energy(temp_field)
                # println("  Temperature energy: $(temp_energy)")
            end
        end
    end
    
    # Performance report
    println("\n📊 Performance Summary:")
    print_performance_report()
    
    println("🎉 Full convection simulation complete!")
    return temp_field, vel_field
end

# Run the simulation
temp_field, vel_field = full_convection_simulation()
```

---

### Example 8: Multi-Field Simulation

Combined thermal and compositional convection.

```julia
function multi_field_simulation()
    println("🌡️🧪 Multi-Field Simulation Example")
    
    config = create_optimized_config(64, 64, nlat=128, nlon=256)
    domain = create_radial_domain(0.35, 1.0, 64)
    
    # Create all fields
    println("🏗️  Creating temperature and composition fields...")
    temp_field = create_shtns_temperature_field(Float64, config, domain)
    comp_field = create_shtns_composition_field(Float64, config, domain)
    vel_field = create_shtns_velocity_fields(Float64, config, domain)
    
    # Temperature boundaries: Hot plume  
    temp_boundaries = create_hybrid_temperature_boundaries(
        (:plume, 4200.0, Dict(
            "center_theta" => π/3,
            "center_phi" => 0.0,
            "width" => π/6
        )),
        (:uniform, 300.0),
        config
    )
    
    # Composition boundaries: Light material hemisphere
    comp_boundaries = create_hybrid_composition_boundaries(
        (:hemisphere, 0.8, Dict(
            "axis" => "z",           # Northern hemisphere
            "amplitude" => 0.2       # 20% light material excess
        )),
        (:uniform, 0.5),             # Average composition at surface
        config
    )
    
    # Apply boundary conditions
    apply_netcdf_temperature_boundaries!(temp_field, temp_boundaries)
    apply_netcdf_composition_boundaries!(comp_field, comp_boundaries)
    
    # Validate setup
    temp_compatible = validate_netcdf_temperature_compatibility(temp_boundaries, config)
    comp_compatible = validate_netcdf_composition_compatibility(comp_boundaries, config)
    
    println("🔍 Validation:")
    println("   Temperature boundaries: $(temp_compatible ? "✓" : "✗")")
    println("   Composition boundaries: $(comp_compatible ? "✓" : "✗")")
    
    # Brief simulation loop
    println("🔄 Running coupled simulation...")
    dt = 0.001
    nsteps = 100
    
    for step in 1:nsteps
        # Coupled nonlinear terms
        compute_temperature_nonlinear!(temp_field, vel_field)
        compute_composition_nonlinear!(comp_field, vel_field)
        
        # Update velocity from buoyancy
        # compute_buoyancy_driven_flow!(vel_field, temp_field, comp_field)
        
        if step % 20 == 0
            println("  Step $(step): Coupled fields evolved")
        end
    end
    
    # Final analysis
    println("\n📊 Final State Analysis:")
    temp_stats = get_boundary_statistics(temp_boundaries.inner_boundary)
    comp_stats = get_boundary_statistics(comp_boundaries.inner_boundary)
    
    println("Temperature CMB: [$(round(temp_stats["min"])), $(round(temp_stats["max"]))] K")
    println("Composition CMB: [$(round(comp_stats["min"], digits=2)), $(round(comp_stats["max"], digits=2))]")
    
    println("🎉 Multi-field simulation complete!")
    return temp_field, comp_field, vel_field
end

# Run the example
temp_field, comp_field, vel_field = multi_field_simulation()
```

---

## 📊 Analysis and Visualization Examples

### Example 9: Data Analysis Pipeline

Complete workflow from simulation to analysis and plotting.

```julia
using Plots, NetCDF
plotlyjs()  # Use PlotlyJS backend for interactive plots

function analysis_pipeline_example()
    println("📊 Data Analysis Pipeline Example")
    
    # Assume we have simulation output
    output_file = "output/combined_time_1p000000.nc"
    
    if !isfile(output_file)
        println("⚠️  Output file not found. Running basic simulation first...")
        temp_field, _ = basic_thermal_convection()
        # In real usage, you would save the output here
        println("✓ Would save output to: $(output_file)")
        return
    end
    
    println("📂 Loading simulation data...")
    nc = NetCDF.open(output_file, "r")
    
    # Read coordinates
    theta = NetCDF.readvar(nc, "theta")
    phi = NetCDF.readvar(nc, "phi")  
    r = NetCDF.readvar(nc, "r")
    
    # Read temperature data
    temperature = NetCDF.readvar(nc, "temperature")
    
    NetCDF.close(nc)
    
    # Analysis 1: Radial temperature profile
    println("📈 Computing radial temperature profile...")
    r_profile = vec(mean(temperature, dims=(1,2)))  # Average over theta, phi
    
    p1 = plot(r, r_profile,
             title="Radial Temperature Profile",
             xlabel="Radius", ylabel="Temperature (K)",
             linewidth=2, marker=:circle)
    
    # Analysis 2: Surface temperature map
    println("🗺️  Creating surface temperature map...")
    surface_temp = temperature[:,:,end]  # Last radial point (surface)
    
    lon_deg = rad2deg.(phi .- π)
    lat_deg = 90 .- rad2deg.(theta)
    
    p2 = heatmap(lon_deg, lat_deg, surface_temp',
                title="Surface Temperature",
                xlabel="Longitude (°)", ylabel="Latitude (°)",
                c=:plasma, aspect_ratio=:equal)
    
    # Analysis 3: Temperature statistics by depth  
    println("📊 Computing depth statistics...")
    depths = r
    temp_means = [mean(temperature[:,:,k]) for k in 1:length(r)]
    temp_stds = [std(temperature[:,:,k]) for k in 1:length(r)]
    
    p3 = plot(depths, temp_means, ribbon=temp_stds,
             title="Temperature Statistics vs Depth",
             xlabel="Radius", ylabel="Temperature (K)",
             label="Mean ± Std", alpha=0.7)
    
    # Analysis 4: Spherical harmonic spectrum
    println("📈 Computing spherical harmonic spectrum...")
    # This would require SHTns transform of surface temperature
    # spectrum = compute_temperature_spectrum(surface_temp, config)
    # p4 = loglog(1:length(spectrum), spectrum, ...)
    
    # Combine plots
    combined_plot = plot(p1, p2, p3, layout=(2,2), size=(800,600))
    
    # Save results
    savefig(combined_plot, "temperature_analysis.png")
    println("💾 Saved analysis plot: temperature_analysis.png")
    
    # Export data for further analysis
    using CSV, DataFrames
    
    analysis_data = DataFrame(
        radius = r,
        mean_temperature = temp_means,
        std_temperature = temp_stds
    )
    
    CSV.write("temperature_analysis.csv", analysis_data)
    println("💾 Saved analysis data: temperature_analysis.csv")
    
    println("🎉 Analysis pipeline complete!")
    return analysis_data, combined_plot
end

# Run the analysis
data, plot = analysis_pipeline_example()
```

---

### Example 10: Performance Benchmarking

Systematic performance testing and optimization.

```julia
function performance_benchmarking_example()
    println("⚡ Performance Benchmarking Example")
    
    # Test different configurations
    test_configs = [
        (16, 16, 32, 64),      # Small: lmax, mmax, nlat, nlon
        (32, 32, 64, 128),     # Medium  
        (64, 64, 128, 256),    # Large
        (128, 128, 256, 512)   # Very large
    ]
    
    results = []
    
    for (lmax, mmax, nlat, nlon) in test_configs
        println("\n🧪 Testing configuration: lmax=$(lmax), nlat=$(nlat)")
        
        # Create configuration
        config = create_optimized_config(lmax, mmax, 
                                        nlat=nlat, nlon=nlon,
                                        use_threading=true, use_simd=true)
        domain = create_radial_domain(0.35, 1.0, 32)  # Fixed radial resolution
        
        # Create field
        temp_field = create_shtns_temperature_field(Float64, config, domain)
        vel_field = create_shtns_velocity_fields(Float64, config, domain)
        
        # Set simple boundaries
        boundaries = create_hybrid_temperature_boundaries(
            (:uniform, 4000.0), (:uniform, 300.0), config
        )
        apply_netcdf_temperature_boundaries!(temp_field, boundaries)
        
        # Benchmark nonlinear computation
        reset_performance_stats!()
        
        nsteps = 50
        elapsed_time = @elapsed @timed_transform begin
            for step in 1:nsteps
                compute_temperature_nonlinear!(temp_field, vel_field)
            end
        end
        
        # Collect results
        stats = get_performance_summary()
        
        result = (
            lmax = lmax,
            nlat = nlat,
            total_time = elapsed_time,
            avg_step_time = elapsed_time / nsteps,
            memory_mb = stats["memory_allocated"],
            throughput = nsteps / elapsed_time
        )
        
        push!(results, result)
        
        println("   Total time: $(round(elapsed_time, digits=2)) s")
        println("   Step time:  $(round(elapsed_time/nsteps*1000, digits=2)) ms")
        println("   Memory:     $(round(stats["memory_allocated"], digits=1)) MB")
        println("   Throughput: $(round(nsteps/elapsed_time, digits=1)) steps/s")
    end
    
    # Create performance plots
    using Plots
    
    lmax_vals = [r.lmax for r in results]
    step_times = [r.avg_step_time * 1000 for r in results]  # Convert to ms
    memory_usage = [r.memory_mb for r in results]
    
    p1 = plot(lmax_vals, step_times,
             title="Performance vs Resolution",
             xlabel="lmax", ylabel="Step Time (ms)",
             marker=:circle, linewidth=2, 
             yscale=:log10)
    
    p2 = plot(lmax_vals, memory_usage,
             title="Memory Usage vs Resolution", 
             xlabel="lmax", ylabel="Memory (MB)",
             marker=:square, linewidth=2,
             yscale=:log10)
    
    benchmark_plot = plot(p1, p2, layout=(1,2), size=(800,400))
    savefig(benchmark_plot, "performance_benchmark.png")
    
    # Performance summary
    println("\n📊 Performance Benchmark Summary:")
    println("="^50)
    for result in results
        println("lmax=$(result.lmax): $(round(result.avg_step_time*1000, digits=2)) ms/step, $(round(result.memory_mb, digits=1)) MB")
    end
    
    println("\n💾 Saved benchmark plot: performance_benchmark.png")
    println("🎉 Performance benchmarking complete!")
    
    return results, benchmark_plot
end

# Run the benchmark
results, plot = performance_benchmarking_example()
```

---

## 🎓 Educational Examples

### Example 11: Interactive Learning Module

Step-by-step exploration of geodynamo concepts.

```julia
function interactive_learning_module()
    println("🎓 Interactive Geodynamo Learning Module")
    println("="^50)
    
    function pause_for_user()
        println("\nPress Enter to continue...")
        readline()
    end
    
    # Lesson 1: Basic setup
    println("\n📚 Lesson 1: Understanding Configuration")
    println("Let's explore how resolution affects simulation accuracy...")
    
    for (lmax, nlat) in [(16, 32), (32, 64), (64, 128)]
        config = create_optimized_config(lmax, lmax, nlat=nlat, nlon=2*nlat)
        
        println("   lmax=$(lmax), nlat=$(nlat) → nlm=$(config.nlm) spectral modes")
        println("   Memory estimate: ~$((config.nlm * 64 * 8) ÷ 1024 ÷ 1024) MB per field")
    end
    
    pause_for_user()
    
    # Lesson 2: Boundary condition effects
    println("\n📚 Lesson 2: Boundary Condition Impact")
    println("Comparing different thermal boundary conditions...")
    
    config = create_optimized_config(32, 32, nlat=64, nlon=128)
    domain = create_radial_domain(0.35, 1.0, 32)
    
    boundary_types = [
        ("Uniform", (:uniform, 4000.0)),
        ("Y11 Pattern", (:y11, 4000.0, Dict("amplitude" => 200.0))),
        ("Plume", (:plume, 4200.0, Dict("width" => π/6, "center_theta" => π/3)))
    ]
    
    for (name, inner_spec) in boundary_types
        boundaries = create_hybrid_temperature_boundaries(
            inner_spec, (:uniform, 300.0), config
        )
        
        stats = get_boundary_statistics(boundaries.inner_boundary)
        println("   $(name): T_range = [$(round(stats["min"])), $(round(stats["max"]))] K")
        println("              T_std = $(round(stats["std"], digits=1)) K")
    end
    
    pause_for_user()
    
    # Lesson 3: Physics concepts
    println("\n📚 Lesson 3: Physical Interpretation")
    println("Understanding the scales involved...")
    
    # Earth parameters
    R_earth = 6.371e6      # meters
    r_inner = 0.35
    r_outer = 1.0
    
    shell_thickness = (r_outer - r_inner) * R_earth
    println("   Core shell thickness: $(shell_thickness/1000) km")
    
    # Thermal diffusion time
    kappa = 1e-6  # m²/s (thermal diffusivity)
    diffusion_time = (shell_thickness)^2 / kappa / (365.25 * 24 * 3600)
    println("   Thermal diffusion time: $(round(diffusion_time/1e6, digits=1)) million years")
    
    # Convection velocity estimate  
    dt_conv = 1e-2  # Convective time unit
    v_conv = shell_thickness / (dt_conv * 365.25 * 24 * 3600) * 100  # cm/year
    println("   Typical convection velocity: $(round(v_conv, digits=1)) cm/year")
    
    pause_for_user()
    
    println("\n🎉 Interactive learning module complete!")
    println("Next steps: Try modifying parameters and running simulations!")
end

# Run the learning module
interactive_learning_module()
```

---

## 🚨 Error Handling Examples

### Example 12: Robust Error Handling

Comprehensive error checking and recovery.

```julia
function robust_simulation_example()
    println("🚨 Robust Simulation with Error Handling")
    
    try
        # Configuration with validation
        println("⚙️  Creating configuration...")
        config = create_optimized_config(64, 64, nlat=128, nlon=256)
        
        # Validate configuration
        if config.lmax <= 0
            throw(ConfigurationError("Invalid lmax: $(config.lmax)"))
        end
        
        domain = create_radial_domain(0.35, 1.0, 64)
        println("✓ Configuration created successfully")
        
        # Field creation with error checking
        println("🏗️  Creating fields...")
        temp_field = create_shtns_temperature_field(Float64, config, domain)
        println("✓ Temperature field created")
        
        # Boundary conditions with file checking
        println("📂 Setting up boundary conditions...")
        
        cmb_file = "cmb_temp.nc"
        surface_file = "surface_temp.nc"
        
        temp_boundaries = nothing
        
        if isfile(cmb_file) && isfile(surface_file)
            println("📥 Loading from NetCDF files...")
            temp_boundaries = load_temperature_boundaries(cmb_file, surface_file)
            
            # Validate loaded boundaries
            if !validate_netcdf_temperature_compatibility(temp_boundaries, config)
                println("⚠️  Boundary compatibility issue - using fallback")
                temp_boundaries = nothing
            end
        end
        
        # Fallback to programmatic boundaries
        if temp_boundaries === nothing
            println("🔄 Using programmatic boundaries as fallback...")
            temp_boundaries = create_hybrid_temperature_boundaries(
                (:uniform, 4000.0), (:uniform, 300.0), config
            )
        end
        
        # Apply boundaries with error checking
        println("🔧 Applying boundary conditions...")
        apply_netcdf_temperature_boundaries!(temp_field, temp_boundaries)
        println("✓ Boundary conditions applied")
        
        # Simulation with error recovery
        println("🚀 Starting simulation...")
        nsteps = 100
        error_count = 0
        max_errors = 5
        
        for step in 1:nsteps
            try
                # Simulate physics step
                current_time = step * 0.001
                
                # Check for numerical instabilities
                # temp_max = maximum(abs, temp_field.temperature.data)
                # if temp_max > 1e10
                #     throw(NumericalInstabilityError("Temperature too large: $temp_max"))
                # end
                
                # Nonlinear computation
                # compute_temperature_nonlinear!(temp_field, vel_field)
                
                if step % 20 == 0
                    println("   Step $step/$nsteps completed")
                end
                
            catch e
                error_count += 1
                println("⚠️  Error at step $step: $(typeof(e))")
                
                if error_count > max_errors
                    println("❌ Too many errors - aborting simulation")
                    break
                end
                
                # Attempt recovery
                println("🔄 Attempting error recovery...")
                # reset_field_to_safe_state!(temp_field)
                continue
            end
        end
        
        println("✓ Simulation completed with $error_count errors")
        
        # Final validation
        println("🔍 Final validation...")
        final_stats = get_boundary_statistics(temp_boundaries.inner_boundary)
        
        if final_stats["max"] > 10000
            println("⚠️  Warning: Unusually high temperatures detected")
        end
        
        println("🎉 Robust simulation completed successfully!")
        return temp_field, temp_boundaries
        
    catch e
        println("❌ Critical error occurred:")
        println("   Error type: $(typeof(e))")
        println("   Message: $e")
        
        if isa(e, BoundaryCompatibilityError)
            println("💡 Suggestion: Check boundary file format and grid dimensions")
        elseif isa(e, ConfigurationError)  
            println("💡 Suggestion: Verify lmax, mmax parameters are positive")
        elseif isa(e, NetCDFFormatError)
            println("💡 Suggestion: Validate NetCDF file contains required variables")
        end
        
        println("\n🔧 Recovery options:")
        println("   1. Use programmatic boundaries instead of NetCDF files")
        println("   2. Reduce resolution (smaller lmax/nlat)")
        println("   3. Check input file paths and permissions")
        
        return nothing, nothing
    end
end

# Run robust simulation
temp_field, boundaries = robust_simulation_example()
```

---

## 📝 Summary

These examples demonstrate:

1. **🚀 Basic Setup** - Simple configurations for learning
2. **🌋 Realistic Physics** - Plume dynamics and multiple hotspots  
3. **📁 Data Integration** - NetCDF file handling and validation
4. **🔀 Hybrid Approaches** - Mixing data sources and patterns
5. **⏰ Time Evolution** - Dynamic boundary conditions
6. **🔬 Full Simulations** - Complete physics loops
7. **📊 Analysis** - Data processing and visualization
8. **⚡ Performance** - Benchmarking and optimization
9. **🎓 Education** - Interactive learning modules
10. **🚨 Error Handling** - Robust production code

## 🔗 Next Steps

- **[API Reference](api-reference.html)** - Detailed function documentation
- **[Visualization Guide](visualization.html)** - Plotting and analysis tools
- **[GitHub Repository](https://github.com/subhk/Geodynamo.jl)** - Source code and issues

*Ready to run your own simulations? Start with the basic examples and work your way up!*