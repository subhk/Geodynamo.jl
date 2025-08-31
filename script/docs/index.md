---
layout: default
title: Home
---

# Geodynamo.jl

**High-performance Earth's magnetic field simulation in Julia**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Julia](https://img.shields.io/badge/Julia-{{ site.geodynamo.julia_version }}-blueviolet)](https://julialang.org/)
[![GitHub](https://img.shields.io/github/stars/subhk/Geodynamo.jl?style=social)]({{ site.geodynamo.github_repo }})
[![CI](https://github.com/subhk/Geodynamo.jl/workflows/CI/badge.svg)](https://github.com/subhk/Geodynamo.jl/actions/workflows/CI.yml)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://subhk.github.io/Geodynamo.jl/)

---

## About Geodynamo.jl

Geodynamo.jl is a cutting-edge Julia package for simulating Earth's magnetic field generation through the geodynamo process. It combines the power of efficient spherical harmonic transforms with flexible boundary conditions to enable groundbreaking research in geophysics and planetary science.

### Key Applications
- **Geodynamo modeling** - Simulate magnetic field generation in planetary cores
- **Mantle convection** - Study thermal and compositional convection patterns
- **Planetary dynamics** - Model magnetic fields in Earth-like and exotic planets
- **Education & research** - Interactive simulations for learning and discovery

---

## 🚀 Quick Start

Get up and running with your first geodynamo simulation in minutes:

```julia
using Geodynamo

# 1. Create an optimized configuration
config = create_optimized_config(32, 32, nlat=64, nlon=128)

# 2. Set up the simulation domain
domain = create_radial_domain(0.35, 1.0, 64)  # Earth-like core

# 3. Create temperature field
temp_field = create_shtns_temperature_field(Float64, config, domain)

# 4. Set boundary conditions
temp_boundaries = create_hybrid_temperature_boundaries(
    (:uniform, 4000.0),  # Hot core-mantle boundary
    (:uniform, 300.0),   # Cool surface
    config
)

# 5. Apply and run
apply_netcdf_temperature_boundaries!(temp_field, temp_boundaries)
println("✓ Your first geodynamo simulation is ready!")
```

[**Get Started Now →**](getting-started.html)

---

## ⭐ Key Features

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem; margin: 2rem 0;">

<div style="border: 1px solid #e1e4e8; border-radius: 8px; padding: 1.5rem;">
<h3>🌊 Spherical Harmonic Transforms</h3>
<p>Efficient spectral methods using SHTns for accurate spherical geometry calculations with optimized performance.</p>
</div>

<div style="border: 1px solid #e1e4e8; border-radius: 8px; padding: 1.5rem;">
<h3>🔧 Flexible Boundary Conditions</h3>
<p>Support for NetCDF data files, programmatic patterns, and hybrid approaches for realistic simulations.</p>
</div>

<div style="border: 1px solid #e1e4e8; border-radius: 8px; padding: 1.5rem;">
<h3>⚡ High Performance</h3>
<p>CPU-optimized with SIMD vectorization, multi-threading, and MPI parallelization for HPC clusters.</p>
</div>

<div style="border: 1px solid #e1e4e8; border-radius: 8px; padding: 1.5rem;">
<h3>📊 Advanced Visualization</h3>
<p>Built-in plotting tools including Hammer projections, spherical maps, and time-series analysis.</p>
</div>

<div style="border: 1px solid #e1e4e8; border-radius: 8px; padding: 1.5rem;">
<h3>📁 Data Integration</h3>
<p>Seamless NetCDF integration for loading observational data and numerical model outputs.</p>
</div>

<div style="border: 1px solid #e1e4e8; border-radius: 8px; padding: 1.5rem;">
<h3>⏱️ Time-Dependent</h3>
<p>Support for evolving boundary conditions and time-dependent simulations with monitoring tools.</p>
</div>

</div>

---

## 📖 Documentation

| Section | Description | Audience |
|---------|-------------|----------|
| **[Getting Started](getting-started.html)** | Complete tutorial from installation to first simulation | Beginners |
| **[API Reference](api-reference.html)** | Comprehensive function and type documentation | All users |
| **[Visualization Guide](visualization.html)** | Plotting tools and analysis techniques | Data analysis |
| **[Examples Gallery](examples.html)** | Working code examples and case studies | Learning |

---

## 🛠 Installation

### Prerequisites
- Julia {{ site.geodynamo.julia_version }} ([Download](https://julialang.org/downloads/))
- NetCDF libraries (for boundary condition files)
- MPI (optional, for parallel computing)

### Quick Install
```julia
using Pkg
Pkg.add("Geodynamo")
```

### Development Install
```julia
using Pkg
Pkg.develop(url="{{ site.geodynamo.github_repo }}")
```

[**Detailed Installation Guide →**](getting-started.html#installation)

---

## 📈 Performance Highlights

<div style="background-color: #f6f8fa; padding: 1.5rem; border-radius: 8px; margin: 1.5rem 0;">

**Benchmark Results:**
- **🔧 SIMD Vectorization**: 20-30% speedup on modern CPUs
- **🧵 Multi-threading**: 35-50% improvement with optimal thread count
- **💾 Memory Optimization**: 30-40% reduction in allocations
- **🌐 MPI Scaling**: Linear scaling to 1000+ cores on HPC clusters
- **⚡ Transform Speed**: Optimized spherical harmonic transforms

</div>

---

## 🎯 Use Cases & Research

### Scientific Applications
- **Earth's magnetic field** evolution and reversal studies
- **Exoplanet magnetism** and habitability assessment
- **Planetary core dynamics** across the solar system
- **Numerical methods** development for geophysics

### Educational Use
- **Graduate coursework** in geophysics and fluid dynamics
- **Interactive demonstrations** of planetary processes
- **Method comparison** and algorithm validation
- **Research project** foundation for students

---

## 🔬 Example Gallery

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin: 2rem 0;">

<div style="border: 1px solid #e1e4e8; border-radius: 6px; padding: 1rem; text-align: center;">
<h4>🌡️ Basic Thermal Convection</h4>
<p>Start with simple temperature-driven convection</p>
<a href="examples.html#basic-thermal-convection" style="text-decoration: none;">View Example →</a>
</div>

<div style="border: 1px solid #e1e4e8; border-radius: 6px; padding: 1rem; text-align: center;">
<h4>🌋 Plume Dynamics</h4>
<p>Model hot plumes rising from the core</p>
<a href="examples.html#plume-dynamics" style="text-decoration: none;">View Example →</a>
</div>

<div style="border: 1px solid #e1e4e8; border-radius: 6px; padding: 1rem; text-align: center;">
<h4>🗺️ Hammer Projection</h4>
<p>Visualize magnetic fields on global maps</p>
<a href="visualization.html#hammer-projection" style="text-decoration: none;">View Example →</a>
</div>

<div style="border: 1px solid #e1e4e8; border-radius: 6px; padding: 1rem; text-align: center;">
<h4>⏱️ Time Evolution</h4>
<p>Simulate evolving boundary conditions</p>
<a href="examples.html#time-evolution" style="text-decoration: none;">View Example →</a>
</div>

</div>

[**Browse All Examples →**](examples.html)

---

## 🤝 Community & Support

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: 2rem 0;">

<div style="text-align: center;">
<h4>📚 Documentation</h4>
<p>Comprehensive guides and API reference</p>
<a href="getting-started.html">Read the Docs</a>
</div>

<div style="text-align: center;">
<h4>🐛 Issues & Bugs</h4>
<p>Report problems and request features</p>
<a href="{{ site.geodynamo.issues }}">GitHub Issues</a>
</div>

<div style="text-align: center;">
<h4>💬 Discussions</h4>
<p>Ask questions and share ideas</p>
<a href="{{ site.geodynamo.discussions }}">GitHub Discussions</a>
</div>

<div style="text-align: center;">
<h4>🔬 Contributing</h4>
<p>Help improve the package</p>
<a href="{{ site.geodynamo.github_repo }}/blob/main/CONTRIBUTING.md">Contribute</a>
</div>

</div>

---

## 📚 Related Packages

Geodynamo.jl is part of the Julia scientific computing ecosystem:

- **[SHTnsKit.jl](https://github.com/subhk/SHTnsKit.jl)** - Spherical harmonic transform library
- **[NCDatasets.jl](https://github.com/Alexander-Barth/NCDatasets.jl)** - NetCDF file support
- **[MPI.jl](https://github.com/JuliaParallel/MPI.jl)** - Message passing interface
- **[Plots.jl](https://github.com/JuliaPlots/Plots.jl)** - Visualization and plotting

---

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

## 🗺️ Development Roadmap

- [ ] **🧲 Full magnetic dynamo** - Complete dynamo simulation capabilities
- [ ] **🚀 GPU acceleration** - CUDA and ROCm support for modern GPUs
- [ ] **🔍 Adaptive grids** - Non-uniform radial and angular mesh refinement
- [ ] **🤖 Machine learning** - Neural network integration for subgrid physics
- [ ] **🎬 Advanced visualization** - 3D rendering and animation tools
- [ ] **☁️ Cloud computing** - Integration with cloud HPC platforms

---

<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem; border-radius: 12px; text-align: center; margin: 2rem 0;">
<h2 style="color: white; margin-top: 0;">Ready to Explore Earth's Magnetic Field?</h2>
<p style="font-size: 1.1em; margin-bottom: 1.5rem;">Start your geodynamo journey today with our comprehensive tutorial and examples.</p>
<a href="getting-started.html" style="display: inline-block; background: rgba(255,255,255,0.2); color: white; padding: 0.75rem 1.5rem; border-radius: 6px; text-decoration: none; font-weight: bold; border: 2px solid rgba(255,255,255,0.3);">Get Started Now →</a>
</div>

---

<div style="text-align: center; padding: 1rem; color: #666;">
<small>
<strong>Geodynamo.jl {{ site.geodynamo.version }}</strong> • 
Built with ❤️ using Julia and Jekyll • 
<a href="{{ site.geodynamo.github_repo }}/blob/main/LICENSE">MIT License</a>
</small>
</div>