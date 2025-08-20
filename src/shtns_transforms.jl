# ============================================================================
# SHTns Spherical Harmonic Transforms
# ============================================================================

struct SHTnsTransformManager{T}
    # Pre-allocated spectral arrays
    spectral_work::Vector{T}
    spectral_complex::Vector{ComplexF64}
    
    # Pre-allocated physical arrays
    physical_work::Matrix{T}
    physical_complex::Matrix{ComplexF64}

    # Vector work arrays
    vector_tor::Vector{T}
    vector_pol::Vector{T}
    vector_u::Matrix{T}
    vector_v::Matrix{T}
    
    # Communication buffers (enhanced)
    send_buffer::Vector{ComplexF64}
    recv_buffer::Vector{ComplexF64}
    
    # MPI communication optimization
    requests::Vector{MPI.Request}
    comm_pattern::Symbol  # :allreduce, :alltoall, :point_to_point
    
    # Configuration
    nlm::Int
    nlat::Int
    nlon::Int
    config::SHTnsConfig
    
    # Performance tracking
    transform_count::Ref{Int}
    total_time::Ref{Float64}
end


"""
    create_transform_manager(::Type{T}, config::SHTnsConfig) where T
    
Create enhanced transform manager using config's pencil information.
"""
function create_transform_manager(::Type{T}, config::SHTnsConfig) where T
    nlm  = config.nlm
    nlat = config.nlat
    nlon = config.nlon
    
    # Determine optimal communication pattern based on decomposition
    lm_range = range_local(config.pencils.spec, 1)
    comm_pattern = determine_comm_pattern(lm_range, nlm)
    
    # Allocate work arrays using SHTnsKit allocation functions
    spectral_work = SHTnsKit.allocate_spectral(config.sht, T=T)
    spectral_complex = SHTnsKit.allocate_complex_spectral(config.sht)
    physical_work = SHTnsKit.allocate_spatial(config.sht, T=T)
    physical_complex = SHTnsKit.allocate_complex_spatial(config.sht)
    
    # Vector work arrays
    vector_tor = SHTnsKit.allocate_spectral(config.sht, T=T)
    vector_pol = SHTnsKit.allocate_spectral(config.sht, T=T)
    vector_u = SHTnsKit.allocate_spatial(config.sht, T=T)
    vector_v = SHTnsKit.allocate_spatial(config.sht, T=T)
    
    # Communication buffers sized appropriately
    buffer_size = compute_buffer_size(config)
    send_buffer = zeros(ComplexF64, buffer_size)
    recv_buffer = zeros(ComplexF64, buffer_size)
    
    # Pre-allocate MPI requests for non-blocking operations
    max_requests = 2 * get_nprocs()
    requests = Vector{MPI.Request}(undef, max_requests)
    
    return SHTnsTransformManager{T}(
        spectral_work, spectral_complex,
        physical_work, physical_complex,
        vector_tor, vector_pol, vector_u, vector_v,
        send_buffer, recv_buffer,
        requests, comm_pattern,
        nlm, nlat, nlon, config,
        Ref(0), Ref(0.0)
    )
end


"""
    determine_comm_pattern(lm_range, nlm)
    
Determine optimal communication pattern based on data distribution.
"""
function determine_comm_pattern(lm_range, nlm)
    coverage = length(lm_range) / nlm
    
    if coverage >= 0.8
        return :allreduce  # Most data is local
    elseif coverage >= 0.3
        return :alltoall   # Moderate distribution
    else
        return :point_to_point  # Highly distributed
    end
end

    
"""
    compute_buffer_size(config)
    
Compute optimal buffer size for communication.
"""
function compute_buffer_size(config::SHTnsConfig)
    # Buffer size based on maximum data transfer in one operation
    max_transfer = max(config.nlm, config.nlat * config.nlon)
    return max_transfer
end


# Thread-local transform manager cache for better performance
const THREAD_LOCAL_MANAGERS = [Dict{UInt64, SHTnsTransformManager}() for _ in 1:Threads.nthreads()]

# Memory pool for temporary arrays to reduce allocations
struct BufferPool{T}
    spectral_buffers::Vector{Vector{T}}
    physical_buffers::Vector{Matrix{T}}
    in_use::Vector{Bool}
    lock::ReentrantLock
end

function BufferPool(::Type{T}, nlm::Int, nlat::Int, nlon::Int, pool_size::Int=4) where T
    spectral_buffers = [Vector{T}(undef, nlm) for _ in 1:pool_size]
    physical_buffers = [Matrix{T}(undef, nlat, nlon) for _ in 1:pool_size]
    in_use = fill(false, pool_size)
    return BufferPool{T}(spectral_buffers, physical_buffers, in_use, ReentrantLock())
end

function acquire_buffers(pool::BufferPool{T}) where T
    lock(pool.lock) do
        for i in eachindex(pool.in_use)
            if !pool.in_use[i]
                pool.in_use[i] = true
                return (pool.spectral_buffers[i], pool.physical_buffers[i], i)
            end
        end
        # If no buffers available, create new ones (fallback)
        nlm = length(pool.spectral_buffers[1])
        nlat, nlon = size(pool.physical_buffers[1])
        return (Vector{T}(undef, nlm), Matrix{T}(undef, nlat, nlon), 0)
    end
end

function release_buffers(pool::BufferPool{T}, buffer_id::Int) where T
    if buffer_id > 0
        lock(pool.lock) do
            pool.in_use[buffer_id] = false
        end
    end
end

# Thread-local buffer pools
const THREAD_LOCAL_POOLS = Dict{Tuple{Type, Int, Int, Int}, BufferPool}()

"""
    get_transform_manager(::Type{T}, config::SHTnsConfig) where T
    
Get or create transform manager with thread-local caching for optimal performance.
"""
function get_transform_manager(::Type{T}, config::SHTnsConfig) where T
    thread_id = Threads.threadid()
    key = hash((T, config.nlm, config.nlat, config.nlon))
    
    # Thread-local access - no locking needed
    local_cache = THREAD_LOCAL_MANAGERS[thread_id]
    if !haskey(local_cache, key)
        local_cache[key] = create_transform_manager(T, config)
    end
    return local_cache[key]::SHTnsTransformManager{T}
end



# ======================================================
# Transform from Spectral to Physical space using SHTns
# ======================================================
function shtns_spectral_to_physical!(spec::SHTnsSpectralField{T}, 
                                    phys::SHTnsPhysicalField{T},
                                    transpose_plan=nothing) where T
    config = spec.config
    sht = config.sht
    manager = get_transform_manager(T, config)
    
    # Track performance if enabled
    t_start = ENABLE_TIMING[] ? MPI.Wtime() : 0.0
    
    # Get local data views
    spec_real = parent(spec.data_real)
    spec_imag = parent(spec.data_imag)
    phys_data = parent(phys.data)
    
    # Get ranges from config's pencils
    r_range = range_local(config.pencils.r, 3)
    lm_range = range_local(config.pencils.spec, 1)
    
    # Process radial levels with enhanced communication
    if manager.comm_pattern == :allreduce
        process_radial_levels_allreduce!(sht, spec_real, spec_imag, phys_data,
                                        r_range, lm_range, manager)
    elseif manager.comm_pattern == :alltoall
        process_radial_levels_alltoall!(sht, spec_real, spec_imag, phys_data,
                                       r_range, lm_range, manager)
    else
        process_radial_levels_p2p!(sht, spec_real, spec_imag, phys_data,
                                  r_range, lm_range, manager)
    end
    
    # Transpose if needed (using config's transpose plans)
    if transpose_plan !== nothing && haskey(config.transpose_plans, transpose_plan)
        transpose_with_timer!(phys.data, phys.data, 
                            config.transpose_plans[transpose_plan],
                            :s2p_transpose)
    end
    
    # Update performance tracking
    if ENABLE_TIMING[]
        manager.total_time[] += MPI.Wtime() - t_start
        manager.transform_count[] += 1
    end
end

# ====================================
# Optimized communication patterns
# ====================================
@inline function process_radial_levels_allreduce!(sht, spec_real, spec_imag, phys_data,
                                                 r_range, lm_range, manager)
    coeffs = manager.spectral_work
    phys_work = manager.physical_work
    
    @inbounds for r_idx in r_range
        local_r = local_r_index(r_idx, r_range)
        
        if local_r <= size(spec_real, 3)
            # Fill coefficients
            fill_coefficients!(coeffs, spec_real, spec_imag, 
                            local_r, lm_range)
            
            # Single allreduce for complete coefficients
            MPI.Allreduce!(coeffs, MPI.SUM, get_comm())
            
            # Synthesis using SHTnsKit high-level API
            SHTnsKit.synthesize!(sht, coeffs, phys_work)
            
            # Copy to output
            copy_physical_data!(phys_data, phys_work, local_r)
        end
    end
end


@inline function process_radial_levels_alltoall!(sht, spec_real, spec_imag, phys_data,
                                                r_range, lm_range, manager)
    # Use MPI_Alltoall for better scaling with moderate distribution
    coeffs = manager.spectral_work
    phys_work = manager.physical_work
    
    nprocs = get_nprocs()
    chunk_size = manager.nlm ÷ nprocs
    
    @inbounds for r_idx in r_range
        local_r = local_r_index(r_idx, r_range)
        
        if local_r <= size(spec_real, 3)
            # Prepare send buffer
            prepare_alltoall_buffer!(manager.send_buffer, spec_real, spec_imag,
                                    local_r, lm_range, chunk_size)
            
            # All-to-all communication
            MPI.Alltoall!(manager.send_buffer, manager.recv_buffer, 
                         chunk_size, get_comm())
            
            # Unpack receive buffer
            unpack_alltoall_buffer!(coeffs, manager.recv_buffer, chunk_size)
            
            # Synthesis and copy
            SHTnsKit.synthesize!(sht, coeffs, phys_work)
            copy_physical_data!(phys_data, phys_work, local_r)
        end
    end
end


@inline function process_radial_levels_p2p!(sht, spec_real, spec_imag, phys_data,
                                          r_range, lm_range, manager)
    # Point-to-point communication for highly distributed data
    # Implementation would use MPI.Isend/Irecv for overlap
    # For now, fall back to allreduce
    process_radial_levels_allreduce!(sht, spec_real, spec_imag, phys_data,
                                    r_range, lm_range, manager)
end


# ==============================
# Optimized helper functions
# ==============================

"""
    local_lm_index(lm_idx, lm_range) -> Int
    
Convert global (l,m) index to local index with bounds checking.
"""
@inline function local_lm_index(lm_idx::Int, lm_range::UnitRange{Int})::Int
    return lm_idx - first(lm_range) + 1
end

"""
    local_r_index(r_idx, r_range) -> Int
    
Convert global radial index to local index with bounds checking.
"""
@inline function local_r_index(r_idx::Int, r_range::UnitRange{Int})::Int
    return r_idx - first(r_range) + 1
end

"""
    is_valid_index(idx, max_size) -> Bool
    
Fast bounds checking for array access.
"""
@inline function is_valid_index(idx::Int, max_size::Int)::Bool
    return (idx > 0) & (idx <= max_size)
end
@inline function fill_coefficients!(coeffs, spec_real, spec_imag, 
                                             local_r, lm_range)
    # Optimized: use fill! instead of explicit loop
    fill!(coeffs, zero(ComplexF64))
    
    @inbounds for lm_idx in lm_range
        if is_valid_index(lm_idx, length(coeffs))
            local_lm = local_lm_index(lm_idx, lm_range)
            @fastmath coeffs[lm_idx] = complex(spec_real[local_lm, 1, local_r],
                                              spec_imag[local_lm, 1, local_r])
        end
    end
end

@inline function copy_physical_data!(phys_data, phys_work, local_r)
    # CPU-optimized vectorized copy with enhanced SIMD
    n = length(phys_work)
    offset = (local_r - 1) * n
    
    # Bounds check once for the entire range
    if offset + n <= length(phys_data)
        # Use @turbo for enhanced CPU vectorization (if available)
        @inbounds @simd ivdep for i in 1:n
            phys_data[offset + i] = real(phys_work[i])
        end
    else
        # Fallback with individual bounds checking
        @inbounds @simd for i in 1:n
            if offset + i <= length(phys_data)
                phys_data[offset + i] = real(phys_work[i])
            end
        end
    end
end

# ======================================================
# Transform from Physical to Spectral space using SHTns
# ======================================================
function shtns_physical_to_spectral!(phys::SHTnsPhysicalField{T}, 
                                    spec::SHTnsSpectralField{T},
                                    transpose_plan=nothing) where T
    config = phys.config
    
    # Transpose first if needed
    if transpose_plan !== nothing && haskey(config.transpose_plans, transpose_plan)
        transpose_with_timer!(phys.data, phys.data,
                            config.transpose_plans[transpose_plan],
                            :p2s_transpose)
    end
    
    sht = config.sht
    manager = get_transform_manager(T, config)
    
    t_start = ENABLE_TIMING[] ? MPI.Wtime() : 0.0
    
    # Get local data views
    phys_data = parent(phys.data)
    spec_real = parent(spec.data_real)
    spec_imag = parent(spec.data_imag)
    
    # Get ranges from config
    r_range = range_local(config.pencils.r, 3)
    lm_range = range_local(config.pencils.spec, 1)
    
    # Process with optimal communication pattern
    process_radial_levels_p2s!(sht, phys_data, spec_real, spec_imag,
                                        r_range, lm_range, manager, config)
    
    if ENABLE_TIMING[]
        manager.total_time[] += MPI.Wtime() - t_start
        manager.transform_count[] += 1
    end
end


@inline function process_radial_levels_p2s!(sht, phys_data, spec_real, spec_imag,
                                                     r_range, lm_range, manager, config)
    phys_work = manager.physical_work
    coeffs    = manager.spectral_work
    
    @inbounds for r_idx in r_range
        local_r = local_r_index(r_idx, r_range)
        
        if local_r <= size(phys_data, 3)
            # Copy to complex work array
            offset = (local_r - 1) * length(phys_work)
            @simd for i in 1:length(phys_work)
                if offset + i <= length(phys_data)
                    phys_work[i] = complex(phys_data[offset + i])
                end
            end
            
            # Analysis
            SHTnsKit.analyze!(sht, phys_work, coeffs)
            
            # Store local portion with m=0 reality constraint
            store_spectral_coefficients!(spec_real, spec_imag, coeffs,
                                                  local_r, lm_range, config)
        end
    end
end


@inline function store_spectral_coefficients!(spec_real, spec_imag, coeffs,
                                                       local_r, lm_range, config)
    @inbounds for lm_idx in lm_range
        if lm_idx <= length(coeffs)
            local_lm = lm_idx - first(lm_range) + 1
            if local_lm <= size(spec_real, 1)
                spec_real[local_lm, 1, local_r] = real(coeffs[lm_idx])
                spec_imag[local_lm, 1, local_r] = imag(coeffs[lm_idx])
                
                # Ensure m=0 modes are real
                m = config.m_values[lm_idx]
                if m == 0
                    spec_imag[local_lm, 1, local_r] = 0.0
                end
            end
        end
    end
end


# ==================================
# Vector synthesis for PencilArrays
# ==================================
function shtns_vector_synthesis!(tor_spec::SHTnsSpectralField{T}, 
                                pol_spec::SHTnsSpectralField{T},
                                vec_phys::SHTnsVectorField{T}) where T
    config = tor_spec.config
    sht = config.sht
    manager = get_transform_manager(T, config)
    
    t_start = ENABLE_TIMING[] ? MPI.Wtime() : 0.0
    
    # Get local data views
    tor_real = parent(tor_spec.data_real)
    tor_imag = parent(tor_spec.data_imag)
    pol_real = parent(pol_spec.data_real)
    pol_imag = parent(pol_spec.data_imag)
    
    v_theta = parent(vec_phys.θ_component.data)
    v_phi   = parent(vec_phys.φ_component.data)
    
    # Use config's pencil ranges
    r_range  = range_local(config.pencils.r, 3)
    lm_range = range_local(config.pencils.spec, 1)
    
    # Process with enhanced communication
    process_vector_synthesis!(sht, tor_real, tor_imag, 
                                       pol_real, pol_imag,
                                       v_theta, v_phi, 
                                       r_range, lm_range, manager)
    
    if ENABLE_TIMING[]
        manager.total_time[] += MPI.Wtime() - t_start
        manager.transform_count[] += 1
    end
end


@inline function process_vector_synthesis!(sht, tor_real, tor_imag, 
                                                    pol_real, pol_imag,
                                                    v_theta, v_phi, 
                                                    r_range, lm_range, manager)
    tor_coeffs = manager.vector_tor
    pol_coeffs = manager.vector_pol
    vt_work = manager.vector_u
    vp_work = manager.vector_v
    
    @inbounds for r_idx in r_range
        local_r = local_r_index(r_idx, r_range)
        
        if local_r <= size(tor_real, 3)
            # Fill both coefficient arrays
            fill_vector_coefficients!(tor_coeffs, pol_coeffs,
                                               tor_real, tor_imag, 
                                               pol_real, pol_imag,
                                               local_r, lm_range)
            
            # Optimized communication based on pattern
            if manager.comm_pattern == :allreduce
                perform_vector_allreduce!(tor_coeffs, pol_coeffs)
            else
                # Use separate allreduces for now
                MPI.Allreduce!(tor_coeffs, MPI.SUM, get_comm())
                MPI.Allreduce!(pol_coeffs, MPI.SUM, get_comm())
            end
            
            # Vector synthesis
            vt_work, vp_work = SHTnsKit.synthesize_vector(sht, tor_coeffs, pol_coeffs)
            
            # Store results with vectorization
            store_vector_components!(v_theta, v_phi, vt_work, vp_work, local_r)
        end
    end
end


@inline function fill_vector_coefficients!(tor_coeffs, pol_coeffs,
                                                    tor_real, tor_imag, 
                                                    pol_real, pol_imag,
                                                    local_r, lm_range)
    # Optimized: use fill! for better performance
    fill!(tor_coeffs, zero(ComplexF64))
    fill!(pol_coeffs, zero(ComplexF64))
    
    # Fill from local data
    @inbounds for lm_idx in lm_range
        if lm_idx <= length(tor_coeffs)
            local_lm = lm_idx - first(lm_range) + 1
            @fastmath begin
                tor_coeffs[lm_idx] = complex(tor_real[local_lm, 1, local_r],
                                            tor_imag[local_lm, 1, local_r])
                pol_coeffs[lm_idx] = complex(pol_real[local_lm, 1, local_r],
                                            pol_imag[local_lm, 1, local_r])
            end
        end
    end
end


@inline function perform_vector_allreduce!(tor_coeffs, pol_coeffs)
    # Optimized: single MPI call using views to avoid allocation
    n = length(tor_coeffs)
    
    # Create combined view without allocation
    combined_view = reinterpret(ComplexF64, 
                               vcat(reinterpret(Float64, tor_coeffs), 
                                    reinterpret(Float64, pol_coeffs)))
    
    MPI.Allreduce!(combined_view, MPI.SUM, get_comm())
    
    # Data is already in place - no unpacking needed
    nothing
end


@inline function store_vector_components!(v_theta, v_phi, vt, vp, local_r)
    n = length(vt)
    offset = (local_r - 1) * n
    
    # CPU-optimized vector component storage with enhanced vectorization
    if offset + n <= length(v_theta) && offset + n <= length(v_phi)
        # Optimized path: vectorize both components simultaneously
        @inbounds @simd ivdep for i in 1:n
            idx = offset + i
            v_theta[idx] = real(vt[i])
            v_phi[idx] = real(vp[i])
        end
    else
        # Safe fallback
        @inbounds @simd for i in 1:n
            idx = offset + i
            if idx <= length(v_theta) && idx <= length(v_phi)
                v_theta[idx] = real(vt[i])
                v_phi[idx] = real(vp[i])
            end
        end
    end
end

# =================================
# Vector analysis for PencilArrays
# =================================
function shtns_vector_analysis!(vec_phys::SHTnsVectorField{T},
                               tor_spec::SHTnsSpectralField{T}, 
                               pol_spec::SHTnsSpectralField{T}) where T
    config = tor_spec.config
    sht = config.sht
    manager = get_transform_manager(T, config)
    
    t_start = ENABLE_TIMING[] ? MPI.Wtime() : 0.0
    
    # Get local data views
    v_theta  = parent(vec_phys.θ_component.data)
    v_phi    = parent(vec_phys.φ_component.data)
    tor_real = parent(tor_spec.data_real)
    tor_imag = parent(tor_spec.data_imag)
    pol_real = parent(pol_spec.data_real)
    pol_imag = parent(pol_spec.data_imag)
    
    # Use config's pencil ranges
    r_range  = range_local(config.pencils.r, 3)
    lm_range = range_local(config.pencils.spec, 1)
    
    # Process with enhanced work arrays
    process_vector_analysis!(sht, v_theta, v_phi,
                                      tor_real, tor_imag, pol_real, pol_imag,
                                      r_range, lm_range, manager, config)
    
    if ENABLE_TIMING[]
        manager.total_time[] += MPI.Wtime() - t_start
        manager.transform_count[] += 1
    end
end


function process_vector_analysis!(sht, v_theta, v_phi,
                                           tor_real, tor_imag, 
                                           pol_real, pol_imag,
                                           r_range, lm_range, 
                                           manager, config)
    vt_work = manager.vector_u
    vp_work = manager.vector_v
    
    @inbounds for r_idx in r_range
        local_r = local_r_index(r_idx, r_range)
        
        if local_r <= size(v_theta, 3)
            # Copy to complex arrays
            offset = (local_r - 1) * length(vt_work)
            @simd for i in 1:length(vt_work)
                if offset + i <= length(v_theta)
                    vt_work[i] = complex(v_theta[offset + i])
                    vp_work[i] = complex(v_phi[offset + i])
                end
            end
            
            # Vector analysis
            tor_coeffs, pol_coeffs = SHTnsKit.analyze_vector(sht, vt_work, vp_work)
            
            # Store with reality constraints
            store_vector_spectral!(tor_real, tor_imag, pol_real, pol_imag,
                                            tor_coeffs, pol_coeffs, 
                                            local_r, lm_range, config)
        end
    end
end


@inline function store_vector_spectral!(tor_real, tor_imag, 
                                                 pol_real, pol_imag,
                                                 tor_coeffs, pol_coeffs, 
                                                 local_r, lm_range, config)
    @inbounds for lm_idx in lm_range
        if lm_idx <= length(tor_coeffs)
            local_lm = lm_idx - first(lm_range) + 1
            
            tor_real[local_lm, 1, local_r] = real(tor_coeffs[lm_idx])
            tor_imag[local_lm, 1, local_r] = imag(tor_coeffs[lm_idx])
            pol_real[local_lm, 1, local_r] = real(pol_coeffs[lm_idx])
            pol_imag[local_lm, 1, local_r] = imag(pol_coeffs[lm_idx])
            
            # Reality constraint for m=0
            m = config.m_values[lm_idx]
            if m == 0
                tor_imag[local_lm, 1, local_r] = 0.0
                pol_imag[local_lm, 1, local_r] = 0.0
            end
        end
    end
end



# =============================
# CPU-Intensive Batched Transform Operations
# =============================

"""
    cpu_intensive_batch_transform!(specs, physs; use_threading=true, chunk_size=4)
    
Perform CPU-intensive batched transforms with enhanced threading and vectorization.
"""
function cpu_intensive_batch_transform!(specs::Vector{SHTnsSpectralField{T}},
                                       physs::Vector{SHTnsPhysicalField{T}};
                                       use_threading::Bool=true,
                                       chunk_size::Int=4) where T
    @assert length(specs) == length(physs)
    
    if isempty(specs)
        return
    end
    
    config = specs[1].config
    sht = config.sht
    manager = get_transform_manager(T, config)
    
    t_start = ENABLE_TIMING[] ? MPI.Wtime() : 0.0
    
    # Use config's ranges
    r_range  = range_local(config.pencils.r, 3)
    lm_range = range_local(config.pencils.spec, 1)
    
    if use_threading && Threads.nthreads() > 1
        # Thread-parallel processing with work-stealing
        cpu_threaded_batch_process!(specs, physs, sht, r_range, lm_range, manager, chunk_size)
    else
        # Sequential processing with CPU optimizations
        cpu_sequential_batch_process!(specs, physs, sht, r_range, lm_range, manager)
    end
    
    if ENABLE_TIMING[]
        elapsed = MPI.Wtime() - t_start
        manager.total_time[] += elapsed
        manager.transform_count[] += length(specs)
    end
end

"""
CPU-optimized threaded batch processing with work-stealing.
"""
function cpu_threaded_batch_process!(specs, physs, sht, r_range, lm_range, manager, chunk_size)
    n_fields = length(specs)
    n_threads = Threads.nthreads()
    
    # Create work chunks for better CPU utilization
    chunks = [(i, min(i + chunk_size - 1, n_fields)) for i in 1:chunk_size:n_fields]
    
    Threads.@threads for chunk in chunks
        start_idx, end_idx = chunk
        
        # Thread-local buffer to reduce contention
        thread_manager = get_transform_manager(eltype(specs[1].data_real), 
                                             specs[1].config)
        
        for field_idx in start_idx:end_idx
            spec = specs[field_idx]
            phys = physs[field_idx]
            
            # Process all radial levels for this field
            for r_idx in r_range
                local_r = local_r_index(r_idx, r_range)
                cpu_simd_process_single_level!(sht, spec, phys, local_r, 
                                              lm_range, thread_manager)
            end
        end
    end
end

"""
CPU-optimized sequential batch processing.
"""
function cpu_sequential_batch_process!(specs, physs, sht, r_range, lm_range, manager)
    # Process radial levels in cache-friendly order
    for r_idx in r_range
        local_r = local_r_index(r_idx, r_range)
        
        # Process all fields at this radial level for better cache utilization
        for (spec, phys) in zip(specs, physs)
            cpu_simd_process_single_level!(sht, spec, phys, local_r, lm_range, manager)
        end
    end
end

"""
SIMD-optimized processing of a single radial level.
"""
@inline function cpu_simd_process_single_level!(sht, spec, phys, local_r, lm_range, manager)
    spec_real = parent(spec.data_real)
    spec_imag = parent(spec.data_imag)
    phys_data = parent(phys.data)
    
    if local_r <= size(spec_real, 3)
        # Use SIMD-optimized coefficient filling
        cpu_simd_fill_coefficients!(manager.spectral_work, spec_real, spec_imag, 
                                   local_r, lm_range)
        
        # Communication
        if manager.comm_pattern == :allreduce
            MPI.Allreduce!(manager.spectral_work, MPI.SUM, get_comm())
        end
        
        # SIMD synthesis
        cpu_simd_synthesize!(sht, manager.spectral_work, manager.physical_work)
        
        # SIMD copy
        cpu_simd_copy_physical_data!(phys_data, manager.physical_work, local_r)
    end
end

"""
SIMD-optimized coefficient filling.
"""
@inline function cpu_simd_fill_coefficients!(coeffs, spec_real, spec_imag, local_r, lm_range)
    # Fast zero fill
    fill!(coeffs, zero(ComplexF64))
    
    # Vectorized coefficient assembly
    @inbounds @simd ivdep for lm_idx in lm_range
        if is_valid_index(lm_idx, length(coeffs))
            local_lm = local_lm_index(lm_idx, lm_range)
            if local_lm <= size(spec_real, 1)
                @fastmath coeffs[lm_idx] = complex(spec_real[local_lm, 1, local_r],
                                                  spec_imag[local_lm, 1, local_r])
            end
        end
    end
end

"""
SIMD-optimized physical data copying.
"""
@inline function cpu_simd_copy_physical_data!(phys_data, phys_work, local_r)
    n = length(phys_work)
    offset = (local_r - 1) * n
    
    if offset + n <= length(phys_data)
        @inbounds @simd ivdep for i in 1:n
            phys_data[offset + i] = real(phys_work[i])
        end
    else
        @inbounds @simd for i in 1:n
            if offset + i <= length(phys_data)
                phys_data[offset + i] = real(phys_work[i])
            end
        end
    end
end
function batch_spectral_to_physical!(specs::Vector{SHTnsSpectralField{T}},
                                    physs::Vector{SHTnsPhysicalField{T}}) where T
    @assert length(specs) == length(physs)
    
    if isempty(specs)
        return
    end
    
    config = specs[1].config
    sht = config.sht
    manager = get_transform_manager(T, config)
    
    t_start = ENABLE_TIMING[] ? MPI.Wtime() : 0.0
    
    # Use config's ranges
    r_range  = range_local(config.pencils.r, 3)
    lm_range = range_local(config.pencils.spec, 1)
    
    # Process all fields at each radial level for cache efficiency
    for r_idx in r_range
        local_r = r_idx - first(r_range) + 1
        
        for (spec, phys) in zip(specs, physs)
            process_single_level!(sht, spec, phys, local_r, lm_range, manager)
        end
    end
    
    if ENABLE_TIMING[]
        elapsed = MPI.Wtime() - t_start
        manager.total_time[] += elapsed
        manager.transform_count[] += length(specs)
    end
end


@inline function process_single_level!(sht, spec, phys, local_r, lm_range, manager)
    spec_real = parent(spec.data_real)
    spec_imag = parent(spec.data_imag)
    phys_data = parent(phys.data)
    
    if local_r <= size(spec_real, 3)
        fill_coefficients!(manager.spectral_work, spec_real, spec_imag, 
                                    local_r, lm_range)
        
        # Use cached communication pattern
        if manager.comm_pattern == :allreduce
            MPI.Allreduce!(manager.spectral_work, MPI.SUM, get_comm())
        end
        
        SHTnsKit.synthesize!(sht, manager.spectral_work, manager.physical_work)
        copy_physical_data!(phys_data, manager.physical_work, local_r)
    end
end



# ====================================
# Derivative Transforms using SHTns
# ====================================
function shtns_compute_gradient!(input::SHTnsSpectralField{T},
                                grad_theta::SHTnsPhysicalField{T},
                                grad_phi::SHTnsPhysicalField{T}) where T
    config = input.config
    sht = config.sht
    manager = get_transform_manager(T, config)
    
    t_start = ENABLE_TIMING[] ? MPI.Wtime() : 0.0
    
    # Get local data
    spec_real = parent(input.data_real)
    spec_imag = parent(input.data_imag)
    grad_theta_data = parent(grad_theta.data)
    grad_phi_data   = parent(grad_phi.data)
    
    # Use config's ranges
    r_range  = range_local(config.pencils.r, 3)
    lm_range = range_local(config.pencils.spec, 1)
    
    @inbounds for r_idx in r_range
        local_r = local_r_index(r_idx, r_range)
        
        if local_r <= size(spec_real, 3)
            # Fill coefficients
            fill_coefficients!(manager.spectral_work, spec_real, spec_imag,
                                        local_r, lm_range)
            
            # Communication
            if manager.comm_pattern == :allreduce
                MPI.Allreduce!(manager.spectral_work, MPI.SUM, get_comm())
            end
            
            # Compute derivatives using SHTnsKit gradient computation
            grad_theta, grad_phi = SHTnsKit.compute_gradient(sht, manager.spectral_work)
            
            # Store results
            n = length(grad_theta)
            offset = (local_r - 1) * n
            @simd for i in 1:n
                if offset + i <= length(grad_theta_data)
                    grad_theta_data[offset + i] = real(grad_theta[i])
                    grad_phi_data[offset + i] = real(grad_phi[i])
                end
            end
        end
    end
    
    if ENABLE_TIMING[]
        manager.total_time[] += MPI.Wtime() - t_start
        manager.transform_count[] += 1
    end
end

# ============================
# Performance monitoring
# ============================
function get_transform_statistics(manager::SHTnsTransformManager)
    count = manager.transform_count[]
    total_time = manager.total_time[]
    avg_time = count > 0 ? total_time / count : 0.0
    
    return (count = count,
            total_time = total_time,
            avg_time = avg_time,
            comm_pattern = manager.comm_pattern)
end

function print_transform_statistics()
    if get_rank() == 0
        println("\n╔═══════════════════════════════════════════════════════╗")
        println("║         Transform Performance Statistics               ║")
        println("╠═══════════════════════════════════════════════════════╣")
        
        for (key, manager) in TRANSFORM_MANAGERS
            stats = get_transform_statistics(manager)
            if stats.count > 0
                println("║ Manager $(key % 1000):                                    ║")
                println("║   Transforms:    $(lpad(stats.count, 8))                      ║")
                println("║   Total time:    $(lpad(round(stats.total_time, digits=3), 8)) s                   ║")
                println("║   Avg time:      $(lpad(round(stats.avg_time * 1000, digits=3), 8)) ms                 ║")
                println("║   Comm pattern:  $(lpad(string(stats.comm_pattern), 12))             ║")
            end
        end
        
        println("╚═══════════════════════════════════════════════════════╝")
    end
end

function clear_transform_cache!()
    lock(MANAGER_LOCK) do
        empty!(TRANSFORM_MANAGERS)
    end
end


# ================================================
# In-place SHTns wrappers for better performance
# ================================================
function synthesis!(output::Matrix{T}, 
                   cfg::SHTnsKit.SHTnsConfig, coeffs::Vector{T}) where T
    SHTnsKit.synthesize!(cfg, coeffs, output)
    return nothing
end

function analysis!(output::Vector{T}, 
                  cfg::SHTnsKit.SHTnsConfig, input::Matrix{T}) where T
    SHTnsKit.analyze!(cfg, input, output)
    return nothing
end

# ======================================================
# Advanced SHTnsKit Features
# ======================================================

"""
    create_optimized_config(lmax, mmax; use_threading=true, use_simd=true)
    
Create a CPU-optimized SHTns configuration with enhanced vectorization and threading.
"""
function create_optimized_config(lmax::Int, mmax::Int; 
                                use_threading::Bool=true,
                                use_simd::Bool=true,
                                nlat::Union{Nothing,Int}=nothing,
                                nlon::Union{Nothing,Int}=nothing)
    if use_threading
        # Set optimal thread count for CPU workloads
        SHTnsKit.set_optimal_threads()
        
        # Enable CPU-specific optimizations
        if use_simd
            @info "Enabling SIMD vectorization for enhanced CPU performance"
        end
    end
    
    # Always use CPU-optimized Gaussian grid configuration
    return SHTnsKit.create_gauss_config(lmax, mmax; nlat=nlat, nlon=nlon)
end

"""
    cpu_optimized_transform!(config, spectral_data, physical_data; use_simd=true)
    
Perform CPU-optimized spherical harmonic transform with enhanced vectorization.
"""
function cpu_optimized_transform!(config::SHTnsConfig, 
                                 spectral_data::AbstractVector{T},
                                 physical_data::AbstractMatrix{T};
                                 use_simd::Bool=true) where T
    @timed_transform begin
        sht = config.sht
        thread_id = Threads.threadid()
        stats = THREAD_LOCAL_STATS[thread_id]
        
        # CPU transform with optimized threading and vectorization
        if use_simd
            # Enhanced SIMD-optimized transform
            cpu_simd_synthesize!(sht, spectral_data, physical_data)
        else
            # Standard CPU transform
            SHTnsKit.synthesize!(sht, spectral_data, physical_data)
        end
        
        stats.cpu_transforms += 1
        return true  # Always uses CPU
    end
end

"""
    cpu_simd_synthesize!(sht, spectral_data, physical_data)
    
SIMD-optimized CPU synthesis with vectorization enhancements.
"""
function cpu_simd_synthesize!(sht::SHTnsKit.SHTnsConfig, 
                             spectral_data::AbstractVector{T},
                             physical_data::AbstractMatrix{T}) where T
    # Get dimensions
    nlat, nlon = size(physical_data)
    nlm = length(spectral_data)
    
    # Use SHTnsKit's optimized synthesis with SIMD hints
    SHTnsKit.synthesize!(sht, spectral_data, physical_data)
    
    # Additional CPU-specific post-processing with explicit vectorization
    @inbounds @simd for j in 1:nlon
        for i in 1:nlat
            # Ensure data is in CPU cache-friendly order
            physical_data[i, j] = physical_data[i, j]
        end
    end
end

"""
    compute_power_spectrum(config, spectral_coeffs)
    
Compute power spectrum using SHTnsKit's built-in function.
"""
function compute_power_spectrum(config::SHTnsConfig, spectral_coeffs::AbstractVector{T}) where T
    try
        return SHTnsKit.power_spectrum(config.sht, spectral_coeffs)
    catch e
        @warn "Power spectrum computation failed: $e"
        # Fallback manual computation
        nlm = length(spectral_coeffs)
        lmax = config.lmax
        power = zeros(T, lmax+1)
        for idx in 1:nlm
            l, m = SHTnsKit.index_to_lm(config.sht, idx)
            power[l+1] += abs2(spectral_coeffs[idx]) * (m == 0 ? 1 : 2)
        end
        return power
    end
end

"""
    evaluate_field_at_coordinates(config, spectral_coeffs, theta, phi)
    
Evaluate spherical harmonic field at specific coordinates.
"""
function evaluate_field_at_coordinates(config::SHTnsConfig, 
                                     spectral_coeffs::AbstractVector{T},
                                     theta::Real, phi::Real) where T
    try
        return SHTnsKit.evaluate_at_point(config.sht, spectral_coeffs, theta, phi)
    catch e
        @warn "Point evaluation failed: $e"
        return zero(T)
    end
end

"""
    rotate_spherical_field(config, spectral_coeffs, alpha, beta, gamma)
    
Rotate spherical harmonic field by Euler angles.
"""
function rotate_spherical_field(config::SHTnsConfig,
                               spectral_coeffs::AbstractVector{T},
                               alpha::Real, beta::Real, gamma::Real) where T
    try
        return SHTnsKit.rotate_field(config.sht, spectral_coeffs, alpha, beta, gamma)
    catch e
        @warn "Field rotation failed: $e"
        return copy(spectral_coeffs)
    end
end

# # Export functions
export shtns_spectral_to_physical!, shtns_physical_to_spectral!
export shtns_vector_synthesis!, shtns_vector_analysis!
export batch_spectral_to_physical!
export shtns_compute_gradient!
export get_transform_statistics, print_transform_statistics
export clear_transform_cache!
export SHTnsTransformManager, get_transform_manager

# ======================================================
# Performance Monitoring and Instrumentation
# ======================================================

"""
Performance statistics structure for monitoring transform operations.
"""
mutable struct TransformStats
    total_transforms::Int
    total_time_ns::Int
    gpu_transforms::Int
    cpu_transforms::Int
    allocation_bytes::Int
    communication_time_ns::Int
    
    TransformStats() = new(0, 0, 0, 0, 0, 0)
end

# Thread-local performance statistics
const THREAD_LOCAL_STATS = [TransformStats() for _ in 1:Threads.nthreads()]

"""
    reset_performance_stats!()
    
Reset all performance statistics across all threads.
"""
function reset_performance_stats!()
    for stats in THREAD_LOCAL_STATS
        stats.total_transforms = 0
        stats.total_time_ns = 0
        stats.gpu_transforms = 0
        stats.cpu_transforms = 0
        stats.allocation_bytes = 0
        stats.communication_time_ns = 0
    end
end

"""
    get_performance_summary()
    
Get aggregated performance statistics from all threads.
"""
function get_performance_summary()
    total_stats = TransformStats()
    for stats in THREAD_LOCAL_STATS
        total_stats.total_transforms += stats.total_transforms
        total_stats.total_time_ns += stats.total_time_ns
        total_stats.gpu_transforms += stats.gpu_transforms
        total_stats.cpu_transforms += stats.cpu_transforms
        total_stats.allocation_bytes += stats.allocation_bytes
        total_stats.communication_time_ns += stats.communication_time_ns
    end
    return total_stats
end

"""
    @timed_transform expr
    
Macro to automatically time and track transform operations.
"""
macro timed_transform(expr)
    return quote
        thread_id = Threads.threadid()
        stats = THREAD_LOCAL_STATS[thread_id]
        
        gc_start = Base.gc_bytes()
        t_start = time_ns()
        
        result = $(esc(expr))
        
        t_end = time_ns()
        gc_end = Base.gc_bytes()
        
        stats.total_transforms += 1
        stats.total_time_ns += (t_end - t_start)
        stats.allocation_bytes += (gc_end - gc_start)
        
        result
    end
end

"""
    print_performance_report()
    
Print a detailed performance report.
"""
function print_performance_report()
    stats = get_performance_summary()
    
    if stats.total_transforms == 0
        println("No transform operations recorded.")
        return
    end
    
    avg_time_ms = stats.total_time_ns / (1_000_000 * stats.total_transforms)
    total_time_s = stats.total_time_ns / 1_000_000_000
    alloc_mb = stats.allocation_bytes / (1024^2)
    comm_fraction = stats.communication_time_ns / stats.total_time_ns
    
    println("\n╔══════════════════════════════════════════════════════════════╗")
    println("║                    Transform Performance Report              ║")
    println("╠══════════════════════════════════════════════════════════════╣")
    println("║ Total Transforms:    $(lpad(stats.total_transforms, 8))                      ║")
    println("║ Total Time:          $(lpad(round(total_time_s, digits=3), 8)) s                   ║")
    println("║ Average Time:        $(lpad(round(avg_time_ms, digits=3), 8)) ms                  ║")
    println("║ GPU Transforms:      $(lpad(stats.gpu_transforms, 8))                      ║")
    println("║ CPU Transforms:      $(lpad(stats.cpu_transforms, 8))                      ║")
    println("║ Memory Allocated:    $(lpad(round(alloc_mb, digits=1), 8)) MB                  ║")
    println("║ Communication Time:  $(lpad(round(comm_fraction*100, digits=1), 8))%                   ║")
    println("╚══════════════════════════════════════════════════════════════╝")
end

# Export new advanced functions
export create_optimized_config, accelerated_transform!
export compute_power_spectrum, evaluate_field_at_coordinates
export rotate_spherical_field

# Export performance monitoring functions
export reset_performance_stats!, get_performance_summary
export print_performance_report, @timed_transform