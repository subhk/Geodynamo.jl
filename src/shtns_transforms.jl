# ============================================================================
# SHTns Spherical Harmonic Transforms
# ============================================================================

struct SHTnsTransformManager{T}
    # Pre-allocated coefficient arrays
    coeffs_full::Vector{ComplexF64}
    coeffs_work::Vector{ComplexF64}
    
    # Pre-allocated physical arrays
    phys_work::Matrix{ComplexF64}
    phys_real::Matrix{T}

    # Vector work arrays
    vt_work::Matrix{ComplexF64}
    vp_work::Matrix{ComplexF64}
    
    # Communication buffers (optimized)
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
    
Create optimized transform manager using config's pencil information.
"""
function create_transform_manager(::Type{T}, config::SHTnsConfig) where T
    nlm  = config.nlm
    nlat = config.nlat
    nlon = config.nlon
    
    # Determine optimal communication pattern based on decomposition
    lm_range = range_local(config.pencils.spec, 1)
    comm_pattern = determine_comm_pattern(lm_range, nlm)
    
    # Allocate work arrays
    coeffs_full = zeros(ComplexF64, nlm)
    coeffs_work = zeros(ComplexF64, nlm)
    phys_work = zeros(ComplexF64, nlat, nlon)
    phys_real = zeros(T, nlat, nlon)
    vt_work = zeros(ComplexF64, nlat, nlon)
    vp_work = zeros(ComplexF64, nlat, nlon)
    
    # Communication buffers sized appropriately
    buffer_size = compute_buffer_size(config)
    send_buffer = zeros(ComplexF64, buffer_size)
    recv_buffer = zeros(ComplexF64, buffer_size)
    
    # Pre-allocate MPI requests for non-blocking operations
    max_requests = 2 * get_nprocs()
    requests = Vector{MPI.Request}(undef, max_requests)
    
    return SHTnsTransformManager{T}(
        coeffs_full, coeffs_work,
        phys_work, phys_real,
        vt_work, vp_work,
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


# Global transform manager cache with thread safety
const TRANSFORM_MANAGERS = Dict{UInt64, SHTnsTransformManager}()
const MANAGER_LOCK = ReentrantLock()

"""
    get_transform_manager(::Type{T}, config::SHTnsConfig) where T
    
Get or create transform manager with caching and thread safety.
"""
function get_transform_manager(::Type{T}, config::SHTnsConfig) where T
    key = hash((Threads.threadid(), T, config.nlm, config.nlat, config.nlon))
    
    lock(MANAGER_LOCK) do
        if !haskey(TRANSFORM_MANAGERS, key)
            TRANSFORM_MANAGERS[key] = create_transform_manager(T, config)
        end
        return TRANSFORM_MANAGERS[key]
    end
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
    
    # Process radial levels with optimized communication
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
                            "s2p_transpose")
    end
    
    # Update performance tracking
    if ENABLE_TIMING[]
        manager.total_time[] += MPI.Wtime() - t_start
        manager.transform_count[] += 1
    end
end


@inline function process_radial_levels_s2p!(sht, spec_real, spec_imag, phys_data,
                                           r_range, lm_range, manager)
    nlm       = manager.nlm
    coeffs    = manager.coeffs_full
    phys_work = manager.phys_work
    
    @inbounds for r_idx in r_range
        local_r = r_idx - first(r_range) + 1
        
        if local_r <= size(spec_real, 3)
            # Fill coefficients efficiently
            fill_coefficients_from_local!(coeffs, spec_real, spec_imag, 
                                         local_r, lm_range)
            
            # Communication if needed (optimized)
            if manager.needs_allreduce
                MPI.Allreduce!(coeffs, MPI.SUM, get_comm())
            end
            
            # Synthesis with pre-allocated output
            synthesis!(phys_work, sht, coeffs)
            
            # Copy to output with vectorization
            copy_physical_data!(phys_data, phys_work, local_r)
        end
    end
end

@inline function fill_coefficients_from_local!(coeffs, spec_real, spec_imag, 
                                              local_r, lm_range)
    # Zero coefficients first (vectorized)
    @simd for i in eachindex(coeffs)
        coeffs[i] = zero(ComplexF64)
    end
    
    # Fill from local data
    @inbounds @simd for lm_idx in lm_range
        if lm_idx <= length(coeffs)
            local_lm = lm_idx - first(lm_range) + 1
            coeffs[lm_idx] = complex(spec_real[local_lm, 1, local_r],
                                    spec_imag[local_lm, 1, local_r])
        end
    end
end

@inline function copy_physical_data!(phys_data, phys_work, local_r)
    @inbounds @simd for idx in eachindex(phys_work)
        phys_data[idx, local_r] = real(phys_work[idx])
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
                            "p2s_transpose")
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
    phys_work = manager.phys_work
    coeffs    = manager.coeffs_full
    
    @inbounds for r_idx in r_range
        local_r = r_idx - first(r_range) + 1
        
        if local_r <= size(phys_data, 3)
            # Copy to complex work array
            offset = (local_r - 1) * length(phys_work)
            @simd for i in 1:length(phys_work)
                if offset + i <= length(phys_data)
                    phys_work[i] = complex(phys_data[offset + i])
                end
            end
            
            # Analysis
            analysis!(coeffs, sht, phys_work)
            
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
    
    # Process with optimized communication
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
    tor_coeffs = manager.coeffs_full
    pol_coeffs = manager.coeffs_work
    vt_work = manager.vt_work
    vp_work = manager.vp_work
    
    @inbounds for r_idx in r_range
        local_r = r_idx - first(r_range) + 1
        
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
            vt_work, vp_work = vector_synthesis(sht, tor_coeffs, pol_coeffs)
            
            # Store results with vectorization
            store_vector_components!(v_theta, v_phi, vt_work, vp_work, local_r)
        end
    end
end


@inline function fill_vector_coefficients!(tor_coeffs, pol_coeffs,
                                                    tor_real, tor_imag, 
                                                    pol_real, pol_imag,
                                                    local_r, lm_range)
    # Vectorized zero
    @simd for i in eachindex(tor_coeffs)
        tor_coeffs[i] = zero(ComplexF64)
        pol_coeffs[i] = zero(ComplexF64)
    end
    
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
    # Single MPI call for both arrays
    n = length(tor_coeffs)
    combined = vcat(tor_coeffs, pol_coeffs)
    MPI.Allreduce!(combined, MPI.SUM, get_comm())
    
    # Unpack
    @simd for i in 1:n
        tor_coeffs[i] = combined[i]
        pol_coeffs[i] = combined[n+i]
    end
end


@inline function store_vector_components!(v_theta, v_phi, vt, vp, local_r)
    n = length(vt)
    offset = (local_r - 1) * n
    
    @inbounds @simd for i in 1:n
        if offset + i <= length(v_theta)
            v_theta[offset + i] = real(vt[i])
            v_phi[offset + i] = real(vp[i])
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
    
    # Process with optimized work arrays
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
                                r_range, lm_range, vt_work, vp_work, config)

    @inbounds for r_idx in r_range
        local_r = r_idx - first(r_range) + 1
        
        if local_r <= size(v_theta, 3) && local_r <= size(v_phi, 3)
            # Copy velocity components to complex work arrays
            for j in 1:size(vt_work, 2)
                @simd for i in 1:size(vt_work, 1)
                    if i <= size(v_theta, 1) && j <= size(v_theta, 2)
                        vt_work[i, j] = complex(v_theta[i, j, local_r])
                        vp_work[i, j] = complex(v_phi[i, j, local_r])
                    end
                end
            end
            
            # Perform vector analysis
            tor_coeffs, pol_coeffs = vector_analysis(sht, vt_work, vp_work)
            
            # Store spectral coefficients
            store_vector_spectral!(tor_real, tor_imag, pol_real, pol_imag,
                                  tor_coeffs, pol_coeffs, local_r, lm_range, config)
        end
    end
end


@inline function store_vector_spectral!(tor_real, tor_imag, 
                                    pol_real, pol_imag,
                                    tor_coeffs, pol_coeffs, 
                                    local_r, lm_range, config)

    @inbounds @simd for lm_idx in lm_range
        if lm_idx <= length(tor_coeffs)
            local_lm = lm_idx - first(lm_range) + 1
            
            tor_real[local_lm, 1, local_r] = real(tor_coeffs[lm_idx])
            tor_imag[local_lm, 1, local_r] = imag(tor_coeffs[lm_idx])
            pol_real[local_lm, 1, local_r] = real(pol_coeffs[lm_idx])
            pol_imag[local_lm, 1, local_r] = imag(pol_coeffs[lm_idx])
            
            # Ensure m=0 modes are real
            m = config.m_values[lm_idx]
            if m == 0
                tor_imag[local_lm, 1, local_r] = 0.0
                pol_imag[local_lm, 1, local_r] = 0.0
            end
        end
    end
end



# =============================
# Batched Transform Operations
# =============================
function batch_spectral_to_physical!(specs::Vector{SHTnsSpectralField{T}},
                                    physs::Vector{SHTnsPhysicalField{T}}) where T
    # Process multiple fields efficiently
    @assert length(specs) == length(physs)
    
    if isempty(specs)
        return
    end
    
    sht = specs[1].config.sht
    manager = get_transform_manager(T, specs[1].config, specs[1].pencil)
    
    # Process all fields at each radial level
    r_range  = get_local_range(specs[1].pencil, 3)
    lm_range = get_local_range(specs[1].pencil, 1)
    
    for r_idx in r_range
        local_r = r_idx - first(r_range) + 1
        
        for (spec, phys) in zip(specs, physs)
            process_single_level_s2p!(sht, spec, phys, local_r, lm_range, manager)
        end
    end
end


@inline function process_single_level_s2p!(sht, spec, phys, local_r, lm_range, manager)
    spec_real = parent(spec.data_real)
    spec_imag = parent(spec.data_imag)
    phys_data = parent(phys.data)
    
    if local_r <= size(spec_real, 3)
        fill_coefficients_from_local!(manager.coeffs_full, spec_real, spec_imag, 
                                     local_r, lm_range)
        
        if manager.needs_allreduce
            MPI.Allreduce!(manager.coeffs_full, MPI.SUM, get_comm())
        end
        
        synthesis!(manager.phys_work, sht, manager.coeffs_full)
        copy_physical_data!(phys_data, manager.phys_work, local_r)
    end
end



# ====================================
# Derivative Transforms using SHTns
# ====================================
function shtns_compute_gradient!(input::SHTnsSpectralField{T},
                                grad_theta::SHTnsPhysicalField{T},
                                grad_phi::SHTnsPhysicalField{T}) where T
    sht = input.config.sht
    manager = get_transform_manager(T, input.config, input.pencil)
    
    # Get local data
    spec_real = parent(input.data_real)
    spec_imag = parent(input.data_imag)

    grad_theta_data = parent(grad_theta.data)
    grad_phi_data   = parent(grad_phi.data)
    
    # Process radial levels
    r_range  = get_local_range(input.pencil, 3)
    lm_range = get_local_range(input.pencil, 1)
    
    @inbounds for r_idx in r_range
        local_r = r_idx - first(r_range) + 1
        
        if local_r <= size(spec_real, 3)
            # Fill coefficients
            fill_coefficients_from_local!(manager.coeffs_full, spec_real, spec_imag,
                                         local_r, lm_range)
            
            if manager.needs_allreduce
                MPI.Allreduce!(manager.coeffs_full, MPI.SUM, get_comm())
            end
            
            # Compute both derivatives
            dtheta = synthesis_dtheta(sht, manager.coeffs_full)
            dphi   = synthesis_dphi(sht, manager.coeffs_full)
            
            # Store results
            @simd for idx in eachindex(dtheta)
                grad_theta_data[idx, local_r] = real(dtheta[idx])
                grad_phi_data[idx, local_r] = real(dphi[idx])
            end
        end
    end
end


# ================================================
# In-place SHTns wrappers for better performance
# ================================================
function synthesis!(output::Matrix{ComplexF64}, 
                    sht::SHTnsSphere, coeffs::Vector{ComplexF64})

    # In-place synthesis
    result  = synthesis(sht, coeffs)
    output .= result
    return nothing #output
end

function analysis!(output::Vector{ComplexF64}, 
                sht::SHTnsSphere, input::Matrix{ComplexF64})

    # In-place analysis
    result  = analysis(sht, input)
    output .= result
    return nothing #output
end


# # Export functions
# export shtns_spectral_to_physical!, shtns_physical_to_spectral!
# export shtns_vector_synthesis!, shtns_vector_analysis!
# export batch_spectral_to_physical!
# export shtns_spectral_to_physical_async!, wait_transform
# export shtns_compute_gradient!
# export clear_transform_cache!, get_transform_statistics
# export SHTnsTransformManager, get_transform_manager