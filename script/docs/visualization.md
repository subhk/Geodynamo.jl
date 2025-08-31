---
layout: page
title: Visualization Guide
permalink: /visualization/
nav_order: 4
description: "Comprehensive guide to plotting and analyzing Geodynamo.jl simulation results"
---

# Visualization and Plotting Guide

Comprehensive guide to plotting and analyzing Geodynamo.jl simulation results.

---

## 🎨 Overview

Geodynamo.jl provides powerful visualization tools for analyzing simulation results, including:

- **Spherical surface plots** at constant radius
- **Hammer projection maps** for global magnetic field visualization  
- **Cross-sectional slices** through the simulation domain
- **Time series analysis** for dynamic evolution
- **Performance monitoring** plots

---

## 🗺️ Spherical Surface Plotting

### Basic Spherical Plots

Plot quantities on spherical surfaces at constant radius using `plot_sphere_r.jl`:

```bash
# Temperature at r=0.8 (lon-lat heatmap)
julia --project=. script/plot_sphere_r.jl ./output/combined_time_1p000000.nc \
      --quantity=temperature --r=0.8 --out=./sphere_temp_r0p8.png

# Velocity z-component at r=0.6  
julia --project=. script/plot_sphere_r.jl ./output/combined_time_1p000000.nc \
      --quantity=velocity_z --r=0.6 --out=./sphere_uz_r0p6.png

# Radial magnetic field at Earth's surface
julia --project=. script/plot_sphere_r.jl ./output/combined_time_1p000000.nc \
      --quantity=magnetic_r --r=1.0 --out=./sphere_br_surface.png
```

### Available Quantities

| Category | Quantities | Description |
|----------|------------|-------------|
| **Scalars** | `temperature`, `composition` | Thermal and compositional fields |
| **Velocity** | `velocity_r`, `velocity_theta`, `velocity_phi`, `velocity_z` | Fluid motion components |
| **Magnetic** | `magnetic_r`, `magnetic_theta`, `magnetic_phi`, `magnetic_z` | Magnetic field components |

### Plotting Options

```bash
--r=<float>          # Radius for extraction (required)
--quantity=<name>    # Field to plot (required)  
--out=<file>         # Output filename (PNG recommended)
--cmap=<name>        # Colormap (default: viridis)
--title=<string>     # Custom plot title
```

**Popular Colormaps:**
- `viridis` - Good for general scalar fields
- `RdBu_r` - Ideal for magnetic fields (red-blue diverging)
- `plasma` - High contrast for temperature
- `seismic` - Blue-white-red for signed quantities

---

## 🔨 Hammer Projection Maps

### Magnetic Field Visualization

Our specialized Hammer projection plotter (`plot_hammer_magnetic.jl`) creates equal-area global maps perfect for magnetic field visualization:

```bash
# Radial magnetic field at Earth's surface in Hammer projection  
julia --project=. script/plot_hammer_magnetic.jl ./output/combined_time_1p000000.nc \
      --r=1.0 --out=./hammer_br_surface.png

# Magnetic field at core-mantle boundary with custom settings
julia --project=. script/plot_hammer_magnetic.jl ./output/combined_time_1p000000.nc \
      --r=1.0 --out=./hammer_br_cmb.png --cmap=RdBu_r --levels=30
```

### Hammer Projection Features

- **Equal-area preservation** - Accurate representation of magnetic field strength
- **Global coverage** - Full sphere in single projection  
- **Meridian/parallel grid** - Geographic reference lines
- **Contour visualization** - Smooth field representation
- **Customizable levels** - Control contour resolution

### Hammer Plot Options

```bash
--r=<float>          # Radius to extract (required)
--out=<file>         # Output image filename
--cmap=<name>        # Colormap (default: RdBu_r)
--title=<string>     # Plot title
--levels=<int>       # Number of contour levels (default: 20)
```

---

## 📏 Cross-Sectional Slices

### Constant-Z Slices

Extract and visualize 2D slices at constant z-coordinate using `plot_slice_z.jl`:

```bash
# Temperature slice at z=0.2
julia --project=. script/plot_slice_z.jl ./output/combined_time_1p000000.nc \
      --quantity=temperature --z=0.2 --out=./slice_temp_z0p2.png

# Magnetic field slice with Cartesian coordinates
julia --project=. script/plot_slice_z.jl ./output/combined_time_1p000000.nc \
      --quantity=magnetic_r --z=0.1 --plane=xy --out=./slice_br_xy.png
```

### Slice Plot Options

```bash
--z=<float>          # Constant z plane (required)
--quantity=<name>    # Field to plot (required)
--out=<file>         # Output filename
--plane=lonlat|xy    # Coordinate system (default: lonlat)
--cmap=<name>        # Colormap
--title=<string>     # Plot title
```

**Coordinate Systems:**
- `lonlat` - Traditional (θ,φ) heatmap
- `xy` - Cartesian scatter plot projection

---

## 📊 Custom Plotting in Julia

### Basic Plotting Setup

```julia
using Plots, NetCDF, SHTnsKit

# Load simulation data
nc = NetCDF.open("output/combined_time_1p000000.nc", "r")

# Read coordinates and magnetic field
theta = NetCDF.readvar(nc, "theta")
phi = NetCDF.readvar(nc, "phi")  
r = NetCDF.readvar(nc, "r")

# Read spectral magnetic field components
mtor_r = NetCDF.readvar(nc, "magnetic_toroidal_real")
mtor_i = NetCDF.readvar(nc, "magnetic_toroidal_imag")
mpol_r = NetCDF.readvar(nc, "magnetic_poloidal_real")
mpol_i = NetCDF.readvar(nc, "magnetic_poloidal_imag")

NetCDF.close(nc)
```

### Synthesize Physical Fields

```julia
# Create SHTns configuration
lmax = 64; mmax = 64
nlat = 128; nlon = 256
config = SHTnsKit.create_gauss_config(lmax, nlat; mmax=mmax, nlon=nlon)

# Synthesize magnetic field components
function synthesize_magnetic_field(config, mtor_r, mtor_i, mpol_r, mpol_i, r_vec)
    lmax, mmax = config.lmax, config.mmax
    nlat, nlon = config.nlat, config.nlon
    nlm, nr = size(mtor_r)
    
    # Initialize output arrays
    br = zeros(Float64, nlat, nlon, nr)
    bt = zeros(Float64, nlat, nlon, nr) 
    bp = zeros(Float64, nlat, nlon, nr)
    
    # Transform for each radial level
    for k in 1:nr
        # Build spectral coefficients
        T = zeros(ComplexF64, lmax+1, mmax+1)
        S = zeros(ComplexF64, lmax+1, mmax+1)
        
        for i in 1:nlm
            l = l_values[i]; m = m_values[i]
            if l <= lmax && m <= mmax
                T[l+1,m+1] = complex(mtor_r[i,k], mtor_i[i,k])
                S[l+1,m+1] = complex(mpol_r[i,k], mpol_i[i,k])
            end
        end
        
        # Synthesize to physical space
        bt[:,:,k], bp[:,:,k] = SHTnsKit.SHsphtor_to_spat(config, S, T)
        
        # Radial component from poloidal potential
        if r_vec[k] > 0
            Q = zeros(ComplexF64, lmax+1, mmax+1)
            for i in 1:nlm
                l = l_values[i]; m = m_values[i] 
                if l <= lmax && m <= mmax
                    Q[l+1,m+1] = S[l+1,m+1] * (l*(l+1)/r_vec[k])
                end
            end
            br[:,:,k] = SHTnsKit.synthesis(config, Q; real_output=true)
        end
    end
    
    return br, bt, bp
end

# Synthesize the magnetic field
br, bt, bp = synthesize_magnetic_field(config, mtor_r, mtor_i, mpol_r, mpol_i, r)
```

### Create Custom Visualizations

```julia
using Plots, PlotlyJS
plotlyjs()

# Extract surface magnetic field (r = 1.0)
r_surface_idx = argmin(abs.(r .- 1.0))
br_surface = br[:,:,r_surface_idx]

# Create lon-lat heatmap
lon_deg = rad2deg.(phi)
lat_deg = 90 .- rad2deg.(theta)  # Convert colatitude to latitude

heatmap(lon_deg, lat_deg, br_surface',
       title="Radial Magnetic Field at Earth's Surface",
       xlabel="Longitude (°)",
       ylabel="Latitude (°)",
       c=:RdBu_r,
       aspect_ratio=:equal,
       size=(800, 400))

savefig("custom_br_surface.png")
```

### Advanced 3D Visualization

```julia
using Plots, PlotlyJS

# Create 3D magnetic field visualization
function plot_magnetic_field_3d(br, bt, bp, r_vec, theta, phi; r_level=1.0)
    # Find closest radial level
    r_idx = argmin(abs.(r_vec .- r_level))
    
    # Extract magnetic field at this radius
    br_slice = br[:,:,r_idx]
    
    # Convert to Cartesian coordinates  
    nlat, nlon = size(br_slice)
    x = zeros(nlat, nlon)
    y = zeros(nlat, nlon) 
    z = zeros(nlat, nlon)
    
    for i in 1:nlat, j in 1:nlon
        x[i,j] = r_level * sin(theta[i]) * cos(phi[j])
        y[i,j] = r_level * sin(theta[i]) * sin(phi[j])
        z[i,j] = r_level * cos(theta[i])
    end
    
    # Create 3D surface plot
    surface(x, y, z,
           surfacecolor=br_slice,
           title="3D Magnetic Field Visualization",
           c=:RdBu_r,
           camera=(45, 30))
end

plot_magnetic_field_3d(br, bt, bp, r, theta, phi; r_level=1.0)
savefig("magnetic_field_3d.png")
```

---

## 📈 Time Series Analysis

### Temporal Evolution Plots

```julia
# Load multiple time snapshots
time_files = ["output/time_0.5.nc", "output/time_1.0.nc", "output/time_1.5.nc"]
times = [0.5, 1.0, 1.5]

# Extract magnetic energy over time
magnetic_energy = Float64[]

for file in time_files
    nc = NetCDF.open(file, "r")
    
    # Read spectral coefficients
    mtor_r = NetCDF.readvar(nc, "magnetic_toroidal_real")
    mtor_i = NetCDF.readvar(nc, "magnetic_toroidal_imag") 
    mpol_r = NetCDF.readvar(nc, "magnetic_poloidal_real")
    mpol_i = NetCDF.readvar(nc, "magnetic_poloidal_imag")
    
    # Compute magnetic energy
    energy = 0.5 * sum(mtor_r.^2 + mtor_i.^2 + mpol_r.^2 + mpol_i.^2)
    push!(magnetic_energy, energy)
    
    NetCDF.close(nc)
end

# Plot time evolution
plot(times, magnetic_energy,
     title="Magnetic Energy Evolution",
     xlabel="Time",
     ylabel="Magnetic Energy",
     linewidth=2,
     marker=:circle)

savefig("magnetic_energy_evolution.png")
```

### Spectral Analysis

```julia
# Magnetic energy by spherical harmonic degree
function compute_energy_spectrum(mtor_r, mtor_i, mpol_r, mpol_i, l_values)
    lmax = maximum(l_values)
    spectrum = zeros(lmax+1)
    
    for i in 1:length(l_values)
        l = l_values[i]
        energy = 0.5 * (mtor_r[i]^2 + mtor_i[i]^2 + mpol_r[i]^2 + mpol_i[i]^2)
        spectrum[l+1] += energy
    end
    
    return spectrum
end

# Load latest data
nc = NetCDF.open("output/combined_time_1p000000.nc", "r")
l_values = Int.(NetCDF.readvar(nc, "l_values"))
mtor_r = NetCDF.readvar(nc, "magnetic_toroidal_real")[:, end]  # Surface values
mtor_i = NetCDF.readvar(nc, "magnetic_toroidal_imag")[:, end] 
mpol_r = NetCDF.readvar(nc, "magnetic_poloidal_real")[:, end]
mpol_i = NetCDF.readvar(nc, "magnetic_poloidal_imag")[:, end]
NetCDF.close(nc)

# Compute and plot spectrum
spectrum = compute_energy_spectrum(mtor_r, mtor_i, mpol_r, mpol_i, l_values)
degrees = 0:length(spectrum)-1

loglog(degrees[2:end], spectrum[2:end],
      title="Magnetic Energy Spectrum",
      xlabel="Spherical Harmonic Degree l", 
      ylabel="Energy",
      marker=:circle,
      linewidth=2)

savefig("magnetic_spectrum.png")
```

---

## 📊 Performance Visualization

### Performance Monitoring Plots

```julia
using Geodynamo

# Collect performance data during simulation
function run_monitored_simulation(nsteps=100)
    reset_performance_stats!()
    
    step_times = Float64[]
    memory_usage = Float64[]
    
    for step in 1:nsteps
        @timed_transform begin
            # Simulation step (placeholder)
            compute_temperature_nonlinear!(temp_field, vel_field)
        end
        
        # Collect statistics
        stats = get_performance_summary()
        push!(step_times, stats["average_time"])
        push!(memory_usage, stats["memory_allocated"])
    end
    
    return step_times, memory_usage
end

# Plot performance metrics
step_times, memory = run_monitored_simulation(50)

# Create performance dashboard
p1 = plot(step_times, title="Step Timing", ylabel="Time (ms)", xlabel="Step")
p2 = plot(memory, title="Memory Usage", ylabel="Memory (MB)", xlabel="Step")  
p3 = histogram(step_times, title="Timing Distribution", xlabel="Time (ms)")

plot(p1, p2, p3, layout=(3,1), size=(800, 600))
savefig("performance_dashboard.png")
```

---

## 🎨 Styling and Customization

### Color Schemes

```julia
# Scientific color schemes for different field types
magnetic_colors = :RdBu_r      # Red-Blue diverging for magnetic fields
temperature_colors = :plasma    # High contrast for temperature  
velocity_colors = :viridis     # Perceptually uniform for velocities
composition_colors = :cividis  # Colorblind-friendly for composition

# Custom colormap
using Colors
function create_magnetic_colormap()
    colors = [RGB(0.0,0.0,0.8),   # Deep blue (negative)
              RGB(0.8,0.8,1.0),   # Light blue  
              RGB(1.0,1.0,1.0),   # White (zero)
              RGB(1.0,0.8,0.8),   # Light red
              RGB(0.8,0.0,0.0)]   # Deep red (positive)
    return cgrad(colors)
end
```

### Publication-Ready Plots

```julia
# Set publication defaults
default(fontfamily="Computer Modern",
        titlefontsize=14,
        guidefontsize=12,
        tickfontsize=10,
        legendfontsize=10,
        dpi=300,
        size=(800,600))

# Create publication figure
function publication_magnetic_plot(br_surface, theta, phi)
    lon_deg = rad2deg.(phi .- π)  # Center at 0° longitude
    lat_deg = 90 .- rad2deg.(theta)
    
    p = heatmap(lon_deg, lat_deg, br_surface',
               title="Radial Magnetic Field at Earth's Surface",
               xlabel="Longitude (°)",
               ylabel="Latitude (°)", 
               c=:RdBu_r,
               clim=(-maximum(abs, br_surface), maximum(abs, br_surface)),
               aspect_ratio=:equal,
               colorbar_title="B_r (nT)")
    
    # Add continent outlines (if available)
    # plot!(continent_lons, continent_lats, color=:black, alpha=0.3)
    
    return p
end
```

---

## 🛠️ Tips and Best Practices

### Performance Tips

1. **Use appropriate file formats**:
   - PNG for web/screen viewing
   - PDF/SVG for publications  
   - HDF5 for large datasets

2. **Optimize plotting resolution**:
   - Match plot resolution to data resolution
   - Use `dpi=300` for publications
   - Reduce resolution for interactive exploration

3. **Memory management**:
   - Close NetCDF files after reading
   - Clear large arrays when not needed
   - Use `@time` to monitor memory usage

### Visualization Strategies

1. **Choose appropriate projections**:
   - Hammer for global equal-area maps
   - Mollweide for astronomical/planetary contexts
   - Orthographic for hemisphere views

2. **Color scheme selection**:
   - Diverging schemes (RdBu_r) for signed fields
   - Sequential schemes (plasma) for positive quantities
   - Perceptually uniform (viridis) for general use

3. **Contour levels**:
   - Use 20-30 levels for smooth appearance
   - Choose meaningful physical values
   - Consider logarithmic spacing for wide ranges

---

## 📚 Advanced Examples

### Animation Creation

```julia
using Plots, ImageMagick

# Create animation of magnetic field evolution
function create_magnetic_animation(time_files, output_gif="magnetic_evolution.gif")
    anim = @animate for (i, file) in enumerate(time_files)
        # Load data
        nc = NetCDF.open(file, "r")
        # ... load and process magnetic field data ...
        
        # Create frame
        heatmap(lon_deg, lat_deg, br_surface',
               title="Magnetic Field - Time: $(times[i])",
               c=:RdBu_r,
               clim=(-br_max, br_max))
        
        NetCDF.close(nc)
    end
    
    gif(anim, output_gif, fps=2)
end
```

### Interactive Visualization

```julia
using PlotlyJS, WebIO

# Create interactive 3D magnetic field explorer
function interactive_magnetic_explorer()
    # ... load and process data ...
    
    plot = plotlyjs()
    surface(x, y, z,
           surfacecolor=br_surface,
           title="Interactive Magnetic Field",
           c=:RdBu_r)
    
    # Add interactivity
    plot!(camera=(45,30), showaxis=true)
end
```

---

## 🔗 Related Tools

- **[GMT.jl](https://github.com/GenericMappingTools/GMT.jl)** - Advanced geographic plotting
- **[PlotlyJS.jl](https://github.com/JuliaPlots/PlotlyJS.jl)** - Interactive web-based plots  
- **[Makie.jl](https://github.com/JuliaPlots/Makie.jl)** - High-performance scientific visualization
- **[GeoMakie.jl](https://github.com/JuliaPlots/GeoMakie.jl)** - Geographic plotting with Makie

---

*Ready to visualize your geodynamo results? Check out our [Examples](examples.html) for working code!*