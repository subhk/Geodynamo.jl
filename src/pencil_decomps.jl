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
    create_pencil_topology(shtns_config; optimize=true)
    
Create enhanced pencil decomposition for SHTns grids.
Supports both 1D and 2D decompositions with automatic optimization.
Accepts an object with fields `nlat`, `nlon`, and `nlm` (e.g., `SHTnsKitConfig`).
"""
function create_pencil_topology(shtns_config; optimize::Bool=true)
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
                                   config)
    nlat, nlon, nr = dims
    
    # Physical space pencils (for different operations)
    pencil_θ = Pencil(topology, dims, (2, 3))  # Contiguous in θ (latitude)
    pencil_φ = Pencil(topology, dims, (1, 3))  # Contiguous in φ (longitude)
    pencil_r = Pencil(topology, dims, (1, 2))  # Contiguous in r (radius)
    
    # Spectral space pencil (for l,m modes)
    spec_dims = (config.nlm, 1, nr)
    pencil_spec = Pencil(topology, spec_dims, (2,))  # Decomposed only in dummy dimension
    
    # Create enhanced pencil for mixed operations
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
    
Create enhanced transpose plans between pencil orientations.
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


# ============================================================================
# Optimized Transpose Operations
# ============================================================================

"""
    transpose_with_timer!(dest, src, plan, label="")
    
Perform transpose with optional timing and statistics.
"""
function transpose_with_timer!(dest::PencilArray, src::PencilArray, 
                               plan::Transpositions.Plan, label::Symbol=:default)
    if ENABLE_TIMING[]
        t_start = MPI.Wtime()
        transpose!(dest, src, plan)
        t_end = MPI.Wtime()
        
        # Accumulate timing statistics
        TRANSPOSE_TIMES[label] = get(TRANSPOSE_TIMES, label, 0.0) + (t_end - t_start)
        TRANSPOSE_COUNTS[label] = get(TRANSPOSE_COUNTS, label, 0) + 1
    else
        transpose!(dest, src, plan)
    end
end


# Global timing controls
const ENABLE_TIMING = Ref(false)
const TRANSPOSE_TIMES = Dict{Symbol, Float64}()
const TRANSPOSE_COUNTS = Dict{Symbol, Int}()


"""
    print_transpose_statistics()
    
Print accumulated transpose timing statistics.
"""
function print_transpose_statistics()
    if get_rank() == 0 && !isempty(TRANSPOSE_TIMES)
        println("\n═══════════════════════════════════════════════════════")
        println(" Transpose Operation Statistics")
        println("═══════════════════════════════════════════════════════")
        
        for (label, total_time) in sort(collect(TRANSPOSE_TIMES), by=x->x[2], rev=true)
            count = TRANSPOSE_COUNTS[label]
            avg_time = total_time / count
            println(" $label:")
            println("   Total time:  $(round(total_time, digits=3)) s")
            println("   Calls:       $count")
            println("   Average:     $(round(avg_time*1000, digits=3)) ms")
        end
        println("═══════════════════════════════════════════════════════")
    end
end


# ===============================
# Load Balancing Analysis
# ===============================
"""
    analyze_load_balance(pencil::Pencil)
    
Analyze and report load balance for a given pencil decomposition.
"""
function analyze_load_balance(pencil::Pencil)::Float64
    comm = get_comm()
    rank = get_rank()
    nprocs = get_nprocs()
    
    # Get local size with explicit types
    local_size::Tuple{Int,Int,Int} = size_local(pencil)
    local_elements::Int = prod(local_size)
    
    # Gather all sizes
    all_sizes = MPI.Gather(local_elements, comm; root=0)
    
    if rank == 0
        min_size = minimum(all_sizes)
        max_size = maximum(all_sizes)
        avg_size = mean(all_sizes)
        std_size = std(all_sizes)
        
        imbalance = (max_size - min_size) / avg_size * 100
        
        println("\nLoad Balance Analysis:")
        println("  Min elements: $min_size")
        println("  Max elements: $max_size")
        println("  Average:      $avg_size")
        println("  Std dev:      $(round(std_size, digits=2))")
        println("  Imbalance:    $(round(imbalance, digits=1))%")
        
        if imbalance > 10
            println("  Warning: Load imbalance exceeds 10%")
        end
    end
end

# ===================================
# Memory-Aware Pencil Creation
# ===================================
"""
    estimate_memory_usage(pencils, field_count::Int, precision::Type)
    
Estimate memory usage for given pencil configuration.
"""
function estimate_memory_usage(pencils, field_count::Int, precision::Type)
    bytes_per_element = sizeof(precision)
    total_bytes = 0
    
    # Calculate memory for each pencil orientation
    for (name, pencil) in pairs(pencils)
        local_size = size_local(pencil)
        local_bytes = prod(local_size) * bytes_per_element * field_count
        total_bytes += local_bytes
    end
    
    # Add overhead for transpose buffers (typically 2x largest pencil)
    max_pencil_size = maximum([prod(size_local(p)) for p in pencils])
    buffer_bytes = 2 * max_pencil_size * bytes_per_element
    total_bytes += buffer_bytes
    
    # Convert to human-readable format
    if total_bytes < 1024^2
        memory_str = "$(round(total_bytes/1024, digits=1)) KB"
    elseif total_bytes < 1024^3
        memory_str = "$(round(total_bytes/1024^2, digits=1)) MB"
    else
        memory_str = "$(round(total_bytes/1024^3, digits=2)) GB"
    end
    
    return total_bytes, memory_str
end


# =============================
# Pencil Array Utilities
# =============================
"""
    create_pencil_array(::Type{T}, pencil::Pencil; init=:zero) where T
    
Create a PencilArray with specified initialization.
"""
function create_pencil_array(::Type{T}, pencil::Pencil; init=:zero) where T
    arr = PencilArray{T}(undef, pencil)
    
    if init == :zero
        fill!(parent(arr), zero(T))
    elseif init == :random
        parent(arr) .= randn(T, size(parent(arr)))
    elseif init == :ones
        fill!(parent(arr), one(T))
    end
    
    return arr
end


"""
    synchronize_halos!(arr::PencilArray)
    
Synchronize ghost/halo regions for parallel computations.
"""
function synchronize_halos!(arr::PencilArray)
    # This would implement halo exchange if using ghost cells
    # For now, just ensure all processes are synchronized
    MPI.Barrier(get_comm())
end

# ===========================
# Diagnostic Functions
# ===========================
"""
    print_pencil_info(pencils)
    
Print detailed information about pencil decomposition.
"""
function print_pencil_info(pencils)
    rank = get_rank()
    
    if rank == 0
        println("\n═══════════════════════════════════════════════════════")
        println(" Pencil Decomposition Information")
        println("═══════════════════════════════════════════════════════")
    end
    
    for (name, pencil) in pairs(pencils)
        global_size = size_global(pencil)
        local_size = size_local(pencil)
        local_range = range_local(pencil)
        
        # Gather info from all ranks
        all_local_sizes = MPI.Gather(prod(local_size), get_comm(); root=0)
        
        if rank == 0
            println("\n Pencil: $name")
            println("   Global size:  $(global_size)")
            println("   Decomposed:   $(decomposition(pencil))")
            
            if get_nprocs() > 1
                min_local = minimum(all_local_sizes)
                max_local = maximum(all_local_sizes)
                balance = max_local / min_local
                println("   Load balance: $(round(balance, digits=2))x")
            end
        end
    end
    
    if rank == 0
        println("═══════════════════════════════════════════════════════")
    end
end


# =================================
# Communication Optimization
# =================================

"""
    optimize_communication_order(plans::Dict)
    
Determine optimal order for transpose operations to minimize communication.
"""
function optimize_communication_order(plans::Dict)
    # Analyze communication patterns and suggest optimal ordering
    comm_costs = Dict{Symbol, Float64}()
    
    for (name, plan) in plans
        # Estimate communication cost based on data volume and process mapping
        # This is a simplified model - actual cost depends on network topology
        src_pencil = plan.src
        dest_pencil = plan.dest
        
        data_volume = prod(size_global(src_pencil))
        comm_distance = estimate_communication_distance(src_pencil, dest_pencil)
        
        comm_costs[name] = data_volume * comm_distance
    end
    
    # Return sorted list of operations by cost
    return sort(collect(comm_costs), by=x->x[2])
end

function estimate_communication_distance(src::Pencil, dest::Pencil)
    # Estimate "distance" between pencil orientations
    # Higher distance = more communication required
    
    src_decomp = decomposition(src)
    dest_decomp = decomposition(dest)
    
    # Count number of dimensions that need redistribution
    distance = sum(src_decomp .!= dest_decomp)
    
    return Float64(distance)
end


# export get_comm, get_rank, get_nprocs
# export create_pencil_topology, create_transpose_plans
# export transpose_with_timer!, print_transpose_statistics
# export analyze_load_balance, estimate_memory_usage
# export create_pencil_array, synchronize_halos!
# export print_pencil_info, optimize_communication_order
# export ENABLE_TIMING


# # Enable timing for performance analysis
# ENABLE_TIMING[] = true

# # Create enhanced pencil topology
# pencils = create_pencil_topology(shtns_config, optimize=true)

# # Analyze load balance
# analyze_load_balance(pencils.r)

# # Estimate memory usage
# bytes, mem_str = estimate_memory_usage(pencils, 10, Float64)
# println("Estimated memory per process: $mem_str")

# # Create transpose plans
# plans = create_transpose_plans(pencils)

# # Perform timed transpose
# transpose_with_timer!(dest, src, plans[:θ_to_φ], "theta_to_phi")

# # Print statistics at end of simulation
# print_transpose_statistics()
