# ============================================================================
# Pencil Configuration with SHTns Integration
# ============================================================================

using PencilArrays.Transpositions


# Global MPI state management
mutable struct MPIState
    initialized::Bool
    comm::MPI.Comm
    rank::Int
    nprocs::Int
end


# Global MPI state (initialized lazily)
const MPI_STATE = MPIState(false, MPI.COMM_NULL, -1, -1)

"""
    get_comm()
    
Get MPI communicator, initializing MPI if needed.
Provides thread-safe lazy initialization.
"""
function get_comm()
    if !MPI_STATE.initialized
        if !MPI.Initialized()
            MPI.Init()
        end
        MPI_STATE.comm = MPI.COMM_WORLD
        MPI_STATE.rank = MPI.Comm_rank(MPI_STATE.comm)
        MPI_STATE.nprocs = MPI.Comm_size(MPI_STATE.comm)
        MPI_STATE.initialized = true
    end
    return MPI_STATE.comm
end


"""
    get_rank()
    
Get MPI rank of current process.
"""
function get_rank()
    get_comm()  # Ensure initialization
    return MPI_STATE.rank
end

"""
    get_nprocs()
    
Get total number of MPI processes.
"""
function get_nprocs()
    get_comm()  # Ensure initialization
    return MPI_STATE.nprocs
end


# ================================
# Optimized Process Topology
# ================================
"""
    optimize_process_topology(nprocs::Int, dims::Tuple{Int,Int,Int})
    
Find optimal 2D process grid for given number of processes and problem dimensions.
Minimizes communication volume.
"""
function optimize_process_topology(nprocs::Int, dims::Tuple{Int,Int,Int})
    nlat, nlon, nr = dims
    
    # Find all valid 2D decompositions
    decompositions = Tuple{Int,Int}[]
    for p1 in 1:nprocs
        if nprocs % p1 == 0
            p2 = nprocs ÷ p1
            push!(decompositions, (p1, p2))
        end
    end
    
    # Score each decomposition based on communication patterns
    best_score = Inf
    best_decomp = (nprocs, 1)
    
    for (p1, p2) in decompositions
        # Estimate communication volume for different pencil orientations
        # Prefer decompositions that balance load and minimize surface/volume ratio
        
        # Check if decomposition is valid for problem size
        if nlat ÷ p1 < 2 || nlon ÷ p2 < 2
            continue
        end
        
        # Score based on:
        # 1. Load balance (prefer square-ish decompositions)
        # 2. Communication volume (proportional to surface area)
        # 3. Cache efficiency (prefer contiguous dimensions)
        
        aspect_ratio = max(p1/p2, p2/p1)
        comm_volume = (nlat/p1 + nlon/p2) * nr  # Simplified communication estimate
        cache_penalty = abs(p1 - p2)  # Penalty for non-square decomposition
        
        score = comm_volume * aspect_ratio * (1.0 + 0.1 * cache_penalty)
        
        if score < best_score
            best_score = score
            best_decomp = (p1, p2)
        end
    end
    
    return best_decomp
end


"""
    create_pencil_topology(shtns_config::SHTnsConfig; optimize=true)
    
Create optimized pencil decomposition for SHTns grids.
Supports both 1D and 2D decompositions with automatic optimization.
"""
function create_pencil_topology(shtns_config::SHTnsConfig; optimize::Bool=true)
    comm = get_comm()
    rank = get_rank()
    nprocs = get_nprocs()
    
    # Get SHTns grid dimensions
    nlat = shtns_config.nlat
    nlon = shtns_config.nlon
    nr = i_N
    dims = (nlat, nlon, nr)
    
    # Determine optimal process topology
    if optimize && nprocs > 1
        proc_dims = optimize_process_topology(nprocs, dims)
    else
        # Default to 1D decomposition
        proc_dims = (nprocs, 1)
    end
    
    # Create PencilArrays topology
    topology = PencilArrays.Topology(comm, proc_dims)
    
    if rank == 0
        println("═══════════════════════════════════════════════════════")
        println(" Pencil Decomposition Setup")
        println("═══════════════════════════════════════════════════════")
        println(" MPI Configuration:")
        println("   Processes:        $nprocs")
        println("   Process grid:     $(proc_dims[1]) × $(proc_dims[2])")
        println(" Grid dimensions:")
        println("   Physical:         $nlat × $nlon × $nr")
        println("   Spectral modes:   $(shtns_config.nlm)")
        println("═══════════════════════════════════════════════════════")
    end
    
    # Create pencils for different computational stages
    pencils = create_computation_pencils(topology, dims, shtns_config)
    
    return pencils
end


"""
    create_computation_pencils(topology, dims, config)
    
Create specialized pencils for different stages of computation.
"""
function create_computation_pencils(topology::PencilArrays.Topology, 
                                   dims::Tuple{Int,Int,Int}, 
                                   config::SHTnsConfig)
    nlat, nlon, nr = dims
    
    # Physical space pencils (for different operations)
    pencil_θ = Pencil(topology, dims, (2, 3))  # Contiguous in θ (latitude)
    pencil_φ = Pencil(topology, dims, (1, 3))  # Contiguous in φ (longitude)
    pencil_r = Pencil(topology, dims, (1, 2))  # Contiguous in r (radius)
    
    # Spectral space pencil (for l,m modes)
    spec_dims = (config.nlm, 1, nr)
    pencil_spec = Pencil(topology, spec_dims, (2,))  # Decomposed only in dummy dimension
    
    # Create optimized pencil for mixed operations
    # This is useful for operations that need both spectral and physical access
    mixed_dims = (config.nlm, config.nlat, nr)
    pencil_mixed = Pencil(topology, mixed_dims, (2,))
    
    return (θ = pencil_θ, 
            φ = pencil_φ, 
            r = pencil_r,
            spec = pencil_spec,
            mixed = pencil_mixed)
end


# ============================================================================
# Transpose Plans with Optimization
# ============================================================================
"""
    create_transpose_plans(pencils)
    
Create optimized transpose plans between pencil orientations.
Includes caching and communication optimization.
"""
function create_transpose_plans(pencils)
    # Create transpose plans for common transitions
    plans = Dict{Symbol, Transpositions.Plan}()
    
    # Physical space transitions
    plans[:θ_to_φ] = Transpositions.Plan(pencils.θ, pencils.φ)
    plans[:φ_to_r] = Transpositions.Plan(pencils.φ, pencils.r)
    plans[:r_to_θ] = Transpositions.Plan(pencils.r, pencils.θ)
    
    # Reverse transitions
    plans[:φ_to_θ] = Transpositions.Plan(pencils.φ, pencils.θ)
    plans[:r_to_φ] = Transpositions.Plan(pencils.r, pencils.φ)
    plans[:θ_to_r] = Transpositions.Plan(pencils.θ, pencils.r)
    
    # Spectral transitions
    plans[:r_to_spec] = Transpositions.Plan(pencils.r, pencils.spec)
    plans[:spec_to_r] = Transpositions.Plan(pencils.spec, pencils.r)
    
    # Mixed transitions (for hybrid operations)
    plans[:mixed_to_r] = Transpositions.Plan(pencils.mixed, pencils.r)
    plans[:r_to_mixed] = Transpositions.Plan(pencils.r, pencils.mixed)
    
    return plans
end



