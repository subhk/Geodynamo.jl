# ============================================================================
# Pencil Configuration with SHTns Integration
# ============================================================================

# using MPI
# using PencilArrays
# using PencilFFTs
# using ..Parameters
# using ..SHTnsSetup

# MPI communicator - make it a function to avoid initialization issues
function get_comm()
    if !MPI.Initialized()
        MPI.Init()
    end
    return MPI.COMM_WORLD
end

# Modified pencil decomposition that works with SHTns grids
function create_pencil_topology(shtns_config::SHTnsConfig)
comm = get_comm()
    
    # Get MPI info
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)
    
    # Use SHTns grid dimensions
    nlat = shtns_config.nlat
    nlon = shtns_config.nlon
    nr = i_N
    
    # Create process topology
    # For PencilArrays, we need to decompose across 2 dimensions
    topology = PencilArrays.Topology(comm, (nprocs, 1))  # 1D decomposition initially
    
    if rank == 0
        println("MPI setup: $nprocs processes")
        println("SHTns grid: $nlat × $nlon × $nr")
    end
    
    # Create pencil decomposition with SHTns dimensions
    dims = (nlat, nlon, nr)
    
    # Pencils for different stages of computation
    # Each pencil decomposes different dimensions
    pencil_θ = Pencil(topology, dims, (2, 3))  # Decomposed in φ and r, contiguous in θ
    pencil_φ = Pencil(topology, dims, (1, 3))  # Decomposed in θ and r, contiguous in φ
    pencil_r = Pencil(topology, dims, (1, 2))  # Decomposed in θ and φ, contiguous in r
    
    # Spectral space pencil (for l,m,r)
    spec_dims = (shtns_config.nlm, 1, nr)
    pencil_spec = Pencil(topology, spec_dims, (1,))  # Decomposed only in lm dimension
    
    return pencil_θ, pencil_φ, pencil_r, pencil_spec
end


# Create transpose plans between pencils
function create_transpose_plans(pencil_θ, pencil_φ, pencil_r)
    # Create transpose plans for moving between pencil orientations
    # These handle the MPI communication
    
    plan_θ_to_φ = PencilArrays.Transpositions.Plan(pencil_θ, pencil_φ)
    plan_φ_to_r = PencilArrays.Transpositions.Plan(pencil_φ, pencil_r)
    plan_r_to_φ = PencilArrays.Transpositions.Plan(pencil_r, pencil_φ)
    plan_φ_to_θ = PencilArrays.Transpositions.Plan(pencil_φ, pencil_θ)
    
    return (θ_to_φ = plan_θ_to_φ, 
            φ_to_r = plan_φ_to_r,
            r_to_φ = plan_r_to_φ,
            φ_to_θ = plan_φ_to_θ)
end


# # Create transforms between pencils
# function create_transforms(pencil_θ, pencil_φ, pencil_r, pencil_spec)
#     # Standard pencil transforms
#     transform_θ_to_φ = PencilArray(pencil_θ) => PencilArray(pencil_φ)
#     transform_φ_to_r = PencilArray(pencil_φ) => PencilArray(pencil_r)
#     transform_φ_to_θ = PencilArray(pencil_φ) => PencilArray(pencil_θ)
#     transform_r_to_φ = PencilArray(pencil_r) => PencilArray(pencil_φ)
    
#     # Spectral transforms
#     transform_r_to_spec = PencilArray(pencil_r) => PencilArray(pencil_spec)
#     transform_spec_to_r = PencilArray(pencil_spec) => PencilArray(pencil_r)
    
#     return (θ_to_φ = transform_θ_to_φ, φ_to_r = transform_φ_to_r,
#             φ_to_θ = transform_φ_to_θ, r_to_φ = transform_r_to_φ,
#             r_to_spec = transform_r_to_spec, spec_to_r = transform_spec_to_r)
# end


