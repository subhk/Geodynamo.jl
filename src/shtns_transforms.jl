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
    
    # Communication buffers
    send_buffer::Vector{ComplexF64}
    recv_buffer::Vector{ComplexF64}
    
    # MPI requests for async operations
    requests::Vector{MPI.Request}
    
    # Configuration
    nlm::Int
    nlat::Int
    nlon::Int
    needs_allreduce::Bool
end


function create_transform_manager(::Type{T}, 
                            config::SHTnsConfig, 
                            pencil::Pencil{3}) where T

    nlm  = config.nlm
    nlat = config.nlat
    nlon = config.nlon
    
    # Check if we need communication
    lm_range = get_local_range(pencil, 1)
    needs_allreduce = length(lm_range) < nlm
    
    return SHTnsTransformManager{T}(
        zeros(ComplexF64, nlm),
        zeros(ComplexF64, nlm),
        zeros(ComplexF64, nlat, nlon),
        zeros(T, nlat, nlon),
        zeros(ComplexF64, nlm),
        zeros(ComplexF64, nlm),
        MPI.Request[],
        nlm, nlat, nlon, needs_allreduce
    )
end
    

# Global transform managers (one per thread for thread safety)
const TRANSFORM_MANAGERS = Dict{UInt64, SHTnsTransformManager}()

function get_transform_manager(::Type{T}, config::SHTnsConfig, pencil::Pencil{3}) where T
    thread_id = Threads.threadid()
    key = hash((thread_id, T, config.nlm, config.nlat, config.nlon))
    
    if !haskey(TRANSFORM_MANAGERS, key)
        TRANSFORM_MANAGERS[key] = create_transform_manager(T, config, pencil)
    end
    
    return TRANSFORM_MANAGERS[key]
end

# ======================================================
# Transform from spectral to physical space using SHTns
# ======================================================
function shtns_spectral_to_physical!(spec::SHTnsSpectralField{T}, 
                                    phys::SHTnsPhysicalField{T},
                                    transpose_plan=nothing) where T
    sht = spec.config.sht
    manager = get_transform_manager(T, spec.config, spec.pencil)
    
    # Get local data views once
    spec_real = parent(spec.data_real)
    spec_imag = parent(spec.data_imag)
    phys_data = parent(phys.data)
    
    # Get local ranges once
    r_range  = get_local_range(spec.pencil, 3)
    lm_range = get_local_range(spec.pencil, 1)
    
    # Process radial levels with optimized memory access
    process_radial_levels_s2p!(sht, spec_real, spec_imag, phys_data,
                               r_range, lm_range, manager)
    
    # Transpose if needed
    if transpose_plan !== nothing
        transpose!(phys.data, transpose_plan)
    end
end

@inline function process_radial_levels_s2p!(sht, spec_real, spec_imag, phys_data,
                                           r_range, lm_range, manager)
    nlm = manager.nlm
    coeffs = manager.coeffs_full
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
# Transform from physical to spectral space using SHTns
# ======================================================
function shtns_physical_to_spectral!(phys::SHTnsPhysicalField{T}, 
                                    spec::SHTnsSpectralField{T},
                                    transpose_plan=nothing) where T
    # Transpose first if needed
    if transpose_plan !== nothing
        transpose!(phys.data, transpose_plan)
    end
    
    sht = spec.config.sht
    manager = get_transform_manager(T, spec.config, spec.pencil)
    
    # Get local data views
    phys_data = parent(phys.data)
    spec_real = parent(spec.data_real)
    spec_imag = parent(spec.data_imag)
    
    # Get local ranges
    r_range = get_local_range(phys.pencil, 3)
    lm_range = get_local_range(spec.pencil, 1)
    
    # Process radial levels
    process_radial_levels_p2s!(sht, phys_data, spec_real, spec_imag,
                               r_range, lm_range, manager, spec.config)
end


@inline function process_radial_levels_p2s!(sht, phys_data, spec_real, spec_imag,
                                           r_range, lm_range, manager, config)
    phys_work = manager.phys_work
    coeffs = manager.coeffs_full
    
    @inbounds for r_idx in r_range
        local_r = r_idx - first(r_range) + 1
        
        if local_r <= size(phys_data, 3)
            # Copy to complex work array (vectorized)
            @simd for idx in 1:length(phys_work)
                phys_work[idx] = complex(phys_data[idx, local_r])
            end
            
            # Analysis
            analysis!(coeffs, sht, phys_work)
            
            # Store in local portion
            store_spectral_coefficients!(spec_real, spec_imag, coeffs,
                                        local_r, lm_range, config)
        end
    end
end


@inline function store_spectral_coefficients!(spec_real, spec_imag, coeffs,
                                             local_r, lm_range, config)
    @inbounds @simd for lm_idx in lm_range
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
    sht = tor_spec.config.sht
    manager = get_transform_manager(T, tor_spec.config, tor_spec.pencil)
    
    # Get local data views
    tor_real = parent(tor_spec.data_real)
    tor_imag = parent(tor_spec.data_imag)
    pol_real = parent(pol_spec.data_real)
    pol_imag = parent(pol_spec.data_imag)
    
    v_theta = parent(vec_phys.θ_component.data)
    v_phi = parent(vec_phys.φ_component.data)
    
    # Get local ranges
    r_range = get_local_range(tor_spec.pencil, 3)
    lm_range = get_local_range(tor_spec.pencil, 1)
    
    # Process with dual coefficient arrays
    process_vector_synthesis!(sht, tor_real, tor_imag, 
                            pol_real, pol_imag,
                            v_theta, v_phi, 
                            r_range, lm_range, manager)
end


@inline function process_vector_synthesis!(sht, tor_real, tor_imag, pol_real, pol_imag,
                                          v_theta, v_phi, r_range, lm_range, manager)
    tor_coeffs = manager.coeffs_full
    pol_coeffs = manager.coeffs_work
    
    @inbounds for r_idx in r_range
        local_r = r_idx - first(r_range) + 1
        
        if local_r <= size(tor_real, 3)
            # Fill both coefficient arrays simultaneously
            fill_vector_coefficients!(tor_coeffs, pol_coeffs,
                                     tor_real, tor_imag, pol_real, pol_imag,
                                     local_r, lm_range)
            
            # Single communication for both
            if manager.needs_allreduce
                perform_vector_allreduce!(tor_coeffs, pol_coeffs)
            end
            
            # Vector synthesis
            vt, vp = vector_synthesis(sht, tor_coeffs, pol_coeffs)
            
            # Store results (vectorized)
            store_vector_components!(v_theta, v_phi, vt, vp, local_r)
        end
    end
end


@inline function fill_vector_coefficients!(tor_coeffs, pol_coeffs,
                                         tor_real, tor_imag, pol_real, pol_imag,
                                         local_r, lm_range)
    # Zero both arrays
    @simd for i in eachindex(tor_coeffs)
        tor_coeffs[i] = zero(ComplexF64)
        pol_coeffs[i] = zero(ComplexF64)
    end
    
    # Fill from local data
    @inbounds @simd for lm_idx in lm_range
        if lm_idx <= length(tor_coeffs)
            local_lm = lm_idx - first(lm_range) + 1
            tor_coeffs[lm_idx] = complex(tor_real[local_lm, 1, local_r],
                                        tor_imag[local_lm, 1, local_r])
            pol_coeffs[lm_idx] = complex(pol_real[local_lm, 1, local_r],
                                        pol_imag[local_lm, 1, local_r])
        end
    end
end


@inline function perform_vector_allreduce!(tor_coeffs, pol_coeffs)
    # Combine into single communication
    combined = vcat(tor_coeffs, pol_coeffs)
    MPI.Allreduce!(combined, MPI.SUM, get_comm())
    
    # Split back
    n = length(tor_coeffs)
    tor_coeffs .= combined[1:n]
    pol_coeffs .= combined[n+1:end]
end



# Vector analysis for PencilArrays
function shtns_vector_analysis!(vec_phys::SHTnsVectorField{T},
                               tor_spec::SHTnsSpectralField{T}, 
                               pol_spec::SHTnsSpectralField{T}) where T

    sht = tor_spec.config.sht
    nlm = tor_spec.config.nlm
    
    for r_idx in get_local_range(vec_phys.θ_component.pencil, 3)
        local_r = r_idx - first(get_local_range(vec_phys.θ_component.pencil, 3)) + 1
        
        # Extract local physical data
        v_theta = complex.(parent(vec_phys.θ_component.data)[:, :, local_r])
        v_phi   = complex.(parent(vec_phys.φ_component.data)[:, :, local_r])
        
        # Vector analysis
        tor_coeffs, pol_coeffs = vector_analysis(sht, v_theta, v_phi)
        
        # Store in local spectral coefficients
        lm_range = get_local_range(tor_spec.pencil, 1)
        for lm_idx in lm_range
            if lm_idx <= nlm
                local_lm = lm_idx - first(lm_range) + 1
                
                parent(tor_spec.data_real)[local_lm, 1, local_r] = real(tor_coeffs[lm_idx])
                parent(tor_spec.data_imag)[local_lm, 1, local_r] = imag(tor_coeffs[lm_idx])
                parent(pol_spec.data_real)[local_lm, 1, local_r] = real(pol_coeffs[lm_idx])
                parent(pol_spec.data_imag)[local_lm, 1, local_r] = imag(pol_coeffs[lm_idx])
                
                # Ensure m=0 modes are real
                m = tor_spec.config.m_values[lm_idx]
                if m == 0
                    parent(tor_spec.data_imag)[local_lm, 1, local_r] = zero(T)
                    parent(pol_spec.data_imag)[local_lm, 1, local_r] = zero(T)
                end
            end
        end
    end
end


# export shtns_spectral_to_physical!, shtns_physical_to_spectral!
# export shtns_vector_synthesis!, shtns_vector_analysis!


# export shtns_spectral_to_physical!, shtns_physical_to_spectral!
# export shtns_compute_theta_derivative!, shtns_compute_phi_derivative!
# export shtns_vector_synthesis!, shtns_vector_analysis!

# end
