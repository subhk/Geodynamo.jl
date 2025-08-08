# Parallelism Strategy Reference Guide
## Geodynamo Simulation with SHTns and PencilArrays

## Overview
The parallelization strategy uses domain decomposition with PencilArrays to distribute the computational work across MPI processes. Different dimensions are parallelized for different operations to minimize communication and maximize efficiency.

## Parallelization Directions

### 1. Spectral Space (l,m modes)
```julia
# Spectral pencil decomposition
spec_dims = (config.nlm, 1, nr)
pencil_spec = Pencil(topology, spec_dims, (2,))  # Decomposed in dummy dimension
```
- **Parallelized**: The (l,m) modes are distributed across processes
- **Local**: Each process owns a subset of spectral modes for ALL radial levels
- **Use case**: Spectral operations, radial derivatives in spectral space

### 2. Physical Space - Three Different Pencils

#### a) θ-pencil (Latitude contiguous)
```julia
pencil_θ = Pencil(topology, dims, (2, 3))  # Decomposed in φ and r
```
- **Parallelized**: φ (longitude) and r (radius) dimensions
- **Local/Contiguous**: θ (latitude) dimension
- **Use case**: Operations requiring latitude derivatives, meridional operations

#### b) φ-pencil (Longitude contiguous)
```julia
pencil_φ = Pencil(topology, dims, (1, 3))  # Decomposed in θ and r
```
- **Parallelized**: θ (latitude) and r (radius) dimensions
- **Local/Contiguous**: φ (longitude) dimension
- **Use case**: FFTs in longitude, azimuthal derivatives

#### c) r-pencil (Radius contiguous)
```julia
pencil_r = Pencil(topology, dims, (1, 2))  # Decomposed in θ and φ
```
- **Parallelized**: θ (latitude) and φ (longitude) dimensions
- **Local/Contiguous**: r (radius) dimension
- **Use case**: Radial derivatives, boundary conditions, nonlinear products

## Visual Representation

### Spectral Space Distribution
```
Process 0: modes (l,m) = [(0,0), (1,0), (1,1), ...]
Process 1: modes (l,m) = [(2,0), (2,1), (2,2), ...]
Process 2: modes (l,m) = [(3,0), (3,1), (3,2), ...]
Process 3: modes (l,m) = [(4,0), (4,1), (4,2), ...]
...
```

### Physical Space Distribution (r-pencil example with 4 processes)
```
              θ (latitude)
         ┌────┬────┬────┬────┐
      φ  │ P0 │ P1 │ P0 │ P1 │  Each process has
(longitude)├────┼────┼────┼────┤  ALL radial points
         │ P2 │ P3 │ P2 │ P3 │  for its θ-φ subdomain
         └────┴────┴────┴────┘
           ↓
         [r₁, r₂, ..., rₙ] <- Complete radial direction per process
```

### 2D Process Grid Example (8 processes)
```
For nprocs = 8:
Option 1: 8×1 (1D decomposition)
Option 2: 4×2 (2D decomposition) <- Often better
Option 3: 2×4 (2D decomposition)

The optimizer chooses based on:
- Load balance
- Communication volume
- Cache efficiency
```

## Workflow and Communication Patterns

### Transform Operations

#### Spectral → Physical
1. Start with spectral coefficients distributed by (l,m)
2. **MPI_Allreduce** to gather all modes (communication!)
3. SHTns synthesis to get physical space
4. Result in r-pencil (radius contiguous)

```julia
# Example code flow
shtns_spectral_to_physical!(spec_field, phys_field)
# Inside: MPI.Allreduce!(coeffs, MPI.SUM, comm)
```

#### Physical → Spectral
1. Start with physical data in r-pencil
2. SHTns analysis at each radial level
3. Scatter results to appropriate processes (communication!)
4. End with spectral coefficients distributed by (l,m)

```julia
# Example code flow
shtns_physical_to_spectral!(phys_field, spec_field)
# Inside: analysis! then scatter to owning processes
```

### Derivative Operations

| Operation | Pencil Required | Communication |
|-----------|----------------|---------------|
| Radial derivatives | r-pencil | None (local) |
| Latitude derivatives | θ-pencil | Transpose if not in θ-pencil |
| Longitude derivatives | φ-pencil | Transpose if not in φ-pencil |
| Spectral derivatives (curl, div) | spectral | None (local in spectral space) |

### Nonlinear Term Computation

```julia
# Typical workflow for nonlinear terms (e.g., u × ω)
# 1. Transform to physical space
shtns_vector_synthesis!(u_tor, u_pol, u_physical)  # Communication
shtns_vector_synthesis!(ω_tor, ω_pol, ω_physical)  # Communication

# 2. Compute products locally in physical space
compute_cross_product!(u_physical, ω_physical, nonlinear)  # No communication

# 3. Transform back to spectral
shtns_vector_analysis!(nonlinear, nl_tor, nl_pol)  # Communication
```

## Communication Patterns

### Primary Communication Operations

1. **MPI_Allreduce**
   - Used in SHTns transforms
   - Gathers spectral coefficients from all processes
   - O(log P) communication steps for P processes

2. **MPI_Alltoall**
   - Used for transposes between pencils
   - Redistributes data among all processes
   - O(P) communication volume

3. **Point-to-point** (MPI_Send/Recv)
   - Used for specific data exchanges
   - Optimized for sparse communication patterns

### Communication Optimization Strategies

```julia
# The code automatically selects communication pattern
function determine_comm_pattern(lm_range, nlm)
    coverage = length(lm_range) / nlm
    
    if coverage >= 0.8
        return :allreduce      # Most data is local
    elseif coverage >= 0.3
        return :alltoall       # Moderate distribution
    else
        return :point_to_point # Highly distributed
    end
end
```

## Process Grid Optimization

The code optimizes the 2D process grid based on:

1. **Load Balance**: Equal work distribution
2. **Communication Volume**: Minimize surface-to-volume ratio
3. **Cache Efficiency**: Prefer contiguous memory access

```julia
function optimize_process_topology(nprocs, dims)
    # Scores each possible decomposition
    # Prefers square-ish grids (e.g., 8×8 over 64×1)
    # Considers problem dimensions
    # Returns optimal (p1, p2) process grid
end
```

## Operations Summary Table

| Operation | Pencil Used | Parallelized Dims | Contiguous Dim | Communication |
|-----------|------------|-------------------|----------------|---------------|
| SHTns synthesis | spec → r | (l,m) → (θ,φ) | r | MPI_Allreduce |
| SHTns analysis | r → spec | (θ,φ) → (l,m) | r | MPI_Allreduce |
| Radial derivatives | r-pencil | (θ,φ) | r | None |
| Radial boundary conditions | r-pencil | (θ,φ) | r | None |
| Latitude operations | θ-pencil | (φ,r) | θ | Transpose |
| Longitude FFTs | φ-pencil | (θ,r) | φ | Transpose |
| Nonlinear products | r-pencil | (θ,φ) | r | None |
| Vorticity (spectral) | spec | (l,m) | - | None |
| Pressure solve | spec | (l,m) | - | None |

## Memory Layout Examples

### Spectral Field Storage
```julia
# For each (l,m) mode on local process:
data_real[local_lm, 1, r]  # Real part
data_imag[local_lm, 1, r]  # Imaginary part
# where local_lm ∈ [1, n_local_modes]
#       r ∈ [1, nr]
```

### Physical Field Storage (r-pencil)
```julia
# For each point in local subdomain:
data[θ_local, φ_local, r]
# where θ_local ∈ [1, n_local_theta]
#       φ_local ∈ [1, n_local_phi]
#       r ∈ [1, nr] (complete)
```

## Scalability Considerations

### Strong Scaling
- **Good to**: ~1000 processes for typical resolutions
- **Limited by**: Communication overhead in transforms
- **Optimization**: Use 2D decomposition, minimize transposes

### Weak Scaling
- **Excellent**: Near-linear with problem size
- **Strategy**: Keep local problem size constant
- **Example**: Double resolution → Double processes

### Communication/Computation Ratio
```
For resolution N:
- Computation: O(N³)
- Communication: O(N²)
- Ratio improves with larger N
```

## Best Practices

1. **Choose grid sizes wisely**
   - Powers of 2 or highly composite numbers for FFTs
   - Divisible by process count for load balance

2. **Minimize transposes**
   - Arrange operations to stay in same pencil
   - Batch operations when possible

3. **Use appropriate pencil**
   - Radial operations → r-pencil
   - Spectral operations → spec pencil
   - Minimize switching between pencils

4. **Monitor performance**
   ```julia
   ENABLE_TIMING[] = true
   # Run simulation
   print_transpose_statistics()
   print_transform_statistics()
   ```

5. **Optimize process topology**
   ```julia
   # Let the code choose optimal decomposition
   create_pencil_topology(config, optimize=true)
   ```

## Typical Workflow Example

```julia
# Main time-stepping loop
while time < final_time
    # 1. Nonlinear terms (physical space)
    #    Already in r-pencil from previous step
    compute_nonlinear_terms!(fields)  # Local operations
    
    # 2. Transform to spectral for linear terms
    shtns_vector_analysis!(u_phys, u_tor, u_pol)  # Communication
    
    # 3. Solve implicit terms in spectral space
    solve_diffusion!(u_tor, u_pol)  # Local operations
    apply_coriolis!(u_tor, u_pol)   # Local operations
    
    # 4. Transform back to physical
    shtns_vector_synthesis!(u_tor, u_pol, u_phys)  # Communication
    
    # 5. Apply boundary conditions
    apply_boundary_conditions!(u_phys)  # Local in r-pencil
    
    time += dt
end
```

## Performance Tips

### Communication Hiding
- Use non-blocking MPI operations where possible
- Overlap computation with communication
- Process multiple fields together (batching)

### Cache Optimization
- Access data in contiguous order
- Use pencils that match operation requirements
- Minimize working set size

### Load Balancing
- Monitor with `analyze_load_balance(pencil)`
- Adjust process grid if imbalance > 10%
- Consider different decompositions for different problem sizes

## Debugging Parallel Issues

```julia
# Check load balance
analyze_load_balance(pencils.r)

# Monitor communication
ENABLE_TIMING[] = true
# ... run simulation ...
print_transpose_statistics()

# Verify data distribution
if get_rank() == 0
    print_pencil_info(pencils)
end

# Check memory usage
bytes, mem_str = estimate_memory_usage(pencils, n_fields, Float64)
println("Memory per process: $mem_str")
```