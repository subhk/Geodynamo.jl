# SHTnsKit.jl Integration in Geodynamo.jl

This document describes the comprehensive integration of the local SHTnsKit.jl library into Geodynamo.jl, replacing the previous SHTnsSpheres dependency with enhanced features and performance optimizations.

## Overview

The integration provides:
- **Modern API**: Uses SHTnsKit.jl's high-level, user-friendly API
- **Enhanced Performance**: GPU acceleration, optimized threading, and memory management
- **Advanced Features**: Power spectrum analysis, field rotation, point evaluation
- **Robust Error Handling**: Graceful degradation and comprehensive diagnostics
- **MPI Integration**: Seamless integration with Geodynamo's parallel decomposition

## Key Changes Made

### 1. Project Configuration
- **Updated Project.toml**: Added SHTnsKit.jl as a dependency with local path reference
- **UUID Updated**: Uses correct SHTnsKit.jl UUID (`1b9d0af0-2e90-4fb0-9eae-4a2e2cdf5320`)
- **Path Dependency**: Configured to use `../SHTnsKit.jl` for local development

### 2. Core Type Updates
- **SHTnsConfig Structure**: Now uses `SHTnsKit.SHTnsConfig` handle instead of custom sphere type
- **Transform Manager**: Redesigned with SHTnsKit-compatible array allocation
- **Field Types**: Updated to work with SHTnsKit's allocation functions

### 3. API Modernization

#### Configuration Creation
```julia
# Old approach
sht = SHTnsSphere(lmax, mmax, grid_type=gaussian, nlat=nlat, nlon=nlon)

# New approach  
sht = SHTnsKit.create_gauss_config(lmax, mmax; nlat=nlat, nlon=nlon)
```

#### Transform Functions
```julia
# Old approach
synthesis!(phys_work, sht, coeffs)
analysis!(coeffs, sht, phys_work)

# New approach
SHTnsKit.synthesize!(sht, coeffs, phys_work)
SHTnsKit.analyze!(sht, phys_work, coeffs)
```

#### Vector Transforms
```julia
# Old approach
vt, vp = vector_synthesis(sht, tor_coeffs, pol_coeffs)
tor, pol = vector_analysis(sht, vt_work, vp_work)

# New approach
vt, vp = SHTnsKit.synthesize_vector(sht, tor_coeffs, pol_coeffs)
tor, pol = SHTnsKit.analyze_vector(sht, vt_work, vp_work)
```

#### Gradient Computation
```julia
# Old approach
dtheta = synthesis_dtheta(sht, coeffs)
dphi = synthesis_dphi(sht, coeffs)

# New approach
grad_theta, grad_phi = SHTnsKit.compute_gradient(sht, coeffs)
```

### 4. Advanced Features Added

#### Optimized Configuration
```julia
config = create_optimized_config(lmax, mmax; 
                                use_gpu=true,           # GPU acceleration
                                use_threading=true,     # Optimized threading
                                nlat=nlat, nlon=nlon)
```

#### Accelerated Transforms
```julia
gpu_used = accelerated_transform!(config, spectral_data, physical_data; use_gpu=true)
```

#### Power Spectrum Analysis
```julia
power = compute_power_spectrum(config, spectral_coeffs)
```

#### Point Evaluation
```julia
value = evaluate_field_at_coordinates(config, spectral_coeffs, theta, phi)
```

#### Field Rotation
```julia
rotated_field = rotate_spherical_field(config, spectral_coeffs, alpha, beta, gamma)
```

### 5. Memory Management
- **Smart Allocation**: Uses SHTnsKit's optimized allocation functions
- **Automatic Cleanup**: Proper resource management with `free_config`
- **Buffer Reuse**: Enhanced buffer management for MPI operations

### 6. Error Handling
- **Graceful Degradation**: Falls back to CPU when GPU fails
- **Status Checking**: Comprehensive SHTns library status checks
- **Platform Detection**: Automatic platform compatibility detection

## Files Modified

### Core Files
- `src/Geodynamo.jl` - Main module with updated exports
- `src/shtns_config.jl` - Configuration structure and creation
- `src/shtns_transforms.jl` - Transform functions and advanced features
- `Project.toml` - Dependencies and path configuration

### Configuration Files
- `src/parameters.jl` - Updated parameter documentation
- `config/default_params.jl` - Updated comments
- `config/template_params.jl` - Updated comments

### Physics Modules
- `src/thermal.jl` - Updated SHTnsKit import
- `src/compositional.jl` - Updated SHTnsKit import
- `src/simulation.jl` - Updated status messages

## New Exported Functions

### Advanced Configuration
- `create_optimized_config` - GPU and threading optimization
- `accelerated_transform!` - Hardware-accelerated transforms

### Analysis Tools
- `compute_power_spectrum` - Built-in power spectrum computation
- `evaluate_field_at_coordinates` - Point evaluation
- `rotate_spherical_field` - Field rotation by Euler angles

## Usage Examples

### Basic Usage
```julia
using Geodynamo

# Create enhanced configuration
config = create_shtns_config(optimize_decomp=true, enable_timing=true)

# Use in simulation
state = initialize_shtns_simulation(config)
run_shtns_simulation!(state)
```

### Advanced Features
```julia
using Geodynamo

# GPU-accelerated configuration
config = create_optimized_config(32, 32; use_gpu=true, use_threading=true)

# High-performance transforms
spectral_data = randn(Float64, SHTnsKit.get_nlm(config.sht))
physical_data = zeros(Float64, 64, 128)
gpu_used = accelerated_transform!(config, spectral_data, physical_data; use_gpu=true)

# Analysis
power = compute_power_spectrum(config, spectral_data)
value_at_equator = evaluate_field_at_coordinates(config, spectral_data, π/2, 0.0)
```

### Demonstration Script
Run the comprehensive demo:
```julia
include("examples/shtnskit_integration_demo.jl")
demo_shtnskit_integration()
```

## Compatibility

### Requirements
- Julia ≥ 1.6
- SHTnsKit.jl (local repository)
- SHTns C library (for full functionality)
- Optional: CUDA.jl for GPU acceleration
- Optional: MPI.jl for parallel execution

### Platform Support
- **Linux**: Full support with conda/compiled SHTns
- **macOS**: Full support with conda/compiled SHTns  
- **Windows**: Limited support (depends on SHTns availability)

### Fallback Behavior
- GPU operations fall back to CPU if unavailable
- SHTns functions provide graceful error handling
- Platform detection warns about known issues

## Performance Benefits

1. **Memory Efficiency**: Optimized allocation reduces memory overhead
2. **Threading**: Automatic thread optimization for available hardware
3. **GPU Acceleration**: Optional CUDA support for large-scale problems
4. **Vectorization**: Enhanced SIMD operations in transform loops
5. **Communication**: Improved MPI patterns for parallel operations

## Migration Notes

### From SHTnsSpheres
- Function signatures have changed (see API sections above)
- Type names updated (`SHTnsSphere` → `SHTnsKit.SHTnsConfig`)
- Grid access functions renamed
- Vector transforms use different return patterns

### Backward Compatibility
- Old parameter names preserved where possible
- Wrapper functions maintain some old interfaces
- Error messages provide migration guidance

## Testing

The integration includes comprehensive testing:
- Syntax validation for all API changes
- Mock testing for development without SHTns
- Integration demos for full functionality
- Performance benchmarking capabilities

## Future Enhancements

Planned improvements:
1. **Adaptive Grid Refinement**: Using SHTnsKit's dynamic sizing
2. **Multi-GPU Support**: Distributed GPU computing
3. **Advanced Rotations**: Wigner D-matrix operations
4. **Spectral Filtering**: Built-in filtering operations
5. **Auto-Differentiation**: Gradient computation for optimization

## Support

For issues related to:
- **Geodynamo Integration**: Create issues in Geodynamo.jl repository
- **SHTnsKit Features**: Refer to SHTnsKit.jl documentation
- **SHTns Library**: Consult SHTns official documentation

---

*This integration leverages the full power of SHTnsKit.jl while maintaining compatibility with Geodynamo.jl's simulation framework.*