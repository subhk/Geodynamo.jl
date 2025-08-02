# ============================================================================
# Pencil Configuration with SHTns Integration
# ============================================================================

# using MPI
# using PencilArrays
# using PencilFFTs
# using ..Parameters
# using ..SHTnsSetup

# MPI communicator
const comm = MPI.COMM_WORLD

# Modified pencil decomposition that works with SHTns grids
function create_pencil_topology(shtns_config::SHTnsConfig)
    # Initialize MPI
    MPI.Init()
    
    # Get MPI info
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)
    
    # Use SHTns grid dimensions
    nlat = shtns_config.nlat
    nlon = shtns_config.nlon
    
    # Create 2D process grid
    P_θ = isqrt(nprocs)
    P_φ = nprocs ÷ P_θ
    
    if rank == 0
        println("MPI setup: $nprocs processes, grid: $P_θ × $P_φ")
        println("SHTns grid: $nlat × $nlon × $(i_N)")
    end
    
    # Create pencil decomposition with SHTns dimensions
    dims = (nlat, nlon, i_N)
    
    # Pencils for different stages of computation
    pencil_θ = Pencil(dims, (1,), comm)      # Distributed along θ 
    pencil_φ = Pencil(dims, (2,), comm)      # Distributed along φ
    pencil_r = Pencil(dims, (3,), comm)      # Distributed along r
    
    # Spectral space pencil (for l,m,r)
    spec_dims = (shtns_config.nlm, 1, i_N)
    pencil_spec = Pencil(spec_dims, (3,), comm)  # Distributed along r
    
    return pencil_θ, pencil_φ, pencil_r, pencil_spec
end

# Create transforms between pencils
function create_transforms(pencil_θ, pencil_φ, pencil_r, pencil_spec)
    # Standard pencil transforms
    transform_θ_to_φ = PencilArray(pencil_θ) => PencilArray(pencil_φ)
    transform_φ_to_r = PencilArray(pencil_φ) => PencilArray(pencil_r)
    transform_φ_to_θ = PencilArray(pencil_φ) => PencilArray(pencil_θ)
    transform_r_to_φ = PencilArray(pencil_r) => PencilArray(pencil_φ)
    
    # Spectral transforms
    transform_r_to_spec = PencilArray(pencil_r) => PencilArray(pencil_spec)
    transform_spec_to_r = PencilArray(pencil_spec) => PencilArray(pencil_r)
    
    return (θ_to_φ = transform_θ_to_φ, φ_to_r = transform_φ_to_r,
            φ_to_θ = transform_φ_to_θ, r_to_φ = transform_r_to_φ,
            r_to_spec = transform_r_to_spec, spec_to_r = transform_spec_to_r)
end


