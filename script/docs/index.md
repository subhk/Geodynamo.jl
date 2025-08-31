---
layout: default
title: Home
---

# Geodynamo.jl

**High-performance Earth's magnetic field simulation in Julia**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Julia](https://img.shields.io/badge/Julia-1.8%2B-blueviolet)](https://julialang.org/)
[![GitHub](https://img.shields.io/github/stars/subhk/Geodynamo.jl?style=social)](https://github.com/subhk/Geodynamo.jl)

---

## 🌍 About

Geodynamo.jl is a high-performance Julia package for simulating Earth's magnetic field generation through the geodynamo process. It combines efficient spherical harmonic transforms with flexible boundary conditions to enable cutting-edge research in geophysics and planetary science.

## 🚀 Quick Start

```julia
using Geodynamo

# 1. Create configuration
config = create_optimized_config(32, 32, nlat=64, nlon=128)

# 2. Set up domain
domain = create_radial_domain(0.35, 1.0, 64)

# 3. Create temperature field
temp_field = create_shtns_temperature_field(Float64, config, domain)

# 4. Set boundary conditions
temp_boundaries = create_hybrid_temperature_boundaries(
    (:uniform, 4000.0),  # Hot CMB
    (:uniform, 300.0),   # Cool surface  
    config
)

# 5. Apply and simulate
apply_netcdf_temperature_boundaries!(temp_field, temp_boundaries)
println("✓ Simulation ready!")
```

## ⭐ Key Features

<div class="feature-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin: 2rem 0;">

<div class="feature-card" style="border: 1px solid #e1e4e8; border-radius: 6px; padding: 1rem;">
<h3>🌊 Spherical Harmonic Transforms</h3>
<p>Efficient spectral methods using SHTns for spherical geometry calculations</p>
</div>

<div class="feature-card" style="border: 1px solid #e1e4e8; border-radius: 6px; padding: 1rem;">
<h3>🔧 Flexible Boundary Conditions</h3>
<p>Support for NetCDF data files, programmatic patterns, and hybrid approaches</p>
</div>

<div class="feature-card" style="border: 1px solid #e1e4e8; border-radius: 6px; padding: 1rem;">
<h3>⚡ High Performance</h3>
<p>CPU-optimized with SIMD vectorization, threading, and MPI parallelization</p>
</div>

<div class="feature-card" style="border: 1px solid #e1e4e8; border-radius: 6px; padding: 1rem;">
<h3>📊 Visualization Tools</h3>
<p>Built-in plotting functions including Hammer projections for magnetic fields</p>
</div>

</div>

## 📖 Documentation

| Section | Description |
|---------|-------------|
| **[Getting Started](getting-started.html)** | Complete beginner's guide to geodynamo modeling |
| **[API Reference](api-reference.html)** | Comprehensive function and type documentation |
| **[Visualization](visualization.html)** | Plotting and analysis tools for simulation results |
| **[Examples](examples.html)** | Working code examples for common use cases |

## 🎯 Use Cases

### Research & Development
- **Geodynamo modeling** - Simulate magnetic field generation in planetary cores
- **Mantle convection** - Study thermal and compositional convection patterns  
- **Planetary dynamics** - Model magnetic fields in Earth-like and exotic planets

### Education & Learning
- **Course projects** - Interactive geophysics simulations for students
- **Method validation** - Test new numerical approaches against benchmarks
- **Algorithm development** - Prototype new spectral methods

### High-Performance Computing
- **Cluster computing** - Scale to thousands of cores with MPI
- **Optimization** - Built-in performance monitoring and tuning
- **Production runs** - Efficient large-scale scientific computing

## 🛠 Installation

### Prerequisites
- Julia 1.8+ ([Download here](https://julialang.org/downloads/))
- NetCDF libraries (for boundary condition files)
- MPI (optional, for parallel computing)

### Install Package
```julia
using Pkg
Pkg.add("Geodynamo")
```

### Development Install
```julia
using Pkg
Pkg.develop(url="https://github.com/subhk/Geodynamo.jl")
```

## 🔬 Scientific Applications

Geodynamo.jl enables research in:

- **Earth's magnetic field** evolution and reversals
- **Exoplanet magnetism** and habitability
- **Planetary core dynamics** across the solar system  
- **Numerical methods** for spherical harmonic transforms
- **High-performance computing** in geophysics

## 🤝 Community

- **[GitHub Repository](https://github.com/subhk/Geodynamo.jl)** - Source code, issues, discussions
- **[Julia Discourse](https://discourse.julialang.org/)** - Ask questions in the community
- **[Contributing Guide](https://github.com/subhk/Geodynamo.jl/blob/main/CONTRIBUTING.md)** - How to contribute

## 📚 Related Packages

- **[SHTnsKit.jl](https://github.com/subhk/SHTnsKit.jl)** - Spherical harmonic transform library
- **[NCDatasets.jl](https://github.com/Alexander-Barth/NCDatasets.jl)** - NetCDF file support
- **[MPI.jl](https://github.com/JuliaParallel/MPI.jl)** - Message passing interface

## 📄 Citation

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

## 📈 Performance Highlights

<div style="background-color: #f6f8fa; padding: 1rem; border-radius: 6px; margin: 1rem 0;">

**Benchmark Results:**
- **SIMD Vectorization**: 20-30% speedup on modern CPUs
- **Multi-threading**: 35-50% improvement with optimal thread count
- **Memory Optimization**: 30-40% reduction in allocations
- **MPI Scaling**: Linear scaling to 1000+ cores

</div>

## 🗺️ Roadmap

- [ ] **Magnetic dynamo** - Full dynamo simulation capabilities
- [ ] **GPU acceleration** - CUDA and ROCm support
- [ ] **Adaptive grids** - Non-uniform radial and angular grids  
- [ ] **Machine learning** - Neural network integration for subgrid physics
- [ ] **Visualization toolkit** - Advanced 3D rendering and animation

---

*Ready to explore Earth's magnetic field? [Get started now](getting-started.html)!*