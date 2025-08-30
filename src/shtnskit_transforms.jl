# ============================================================================
# SHTnsKit Spherical Harmonic Transforms
# ============================================================================
# 
# This module implements spherical harmonic transforms using SHTnsKit.jl
# with MPI parallelization across theta and phi directions using PencilArrays
#

using SHTnsKit
using MPI
using PencilArrays
using FFTW
using LinearAlgebra

# ============================================================================
# SHTnsKit Configuration Structure
# ============================================================================

struct SHTnsKitConfig
    # SHTnsKit configuration
    sht_config::SHTnsKit.SHTConfig
    
    # Grid parameters
    nlat::Int
    nlon::Int
    lmax::Int
    mmax::Int
    nlm::Int
    
    # PencilArrays decomposition for parallelization
    pencils::NamedTuple
    
    # Performance optimization
    plan::Union{SHTnsKit.SHTPlan, Nothing}
    
    # Memory estimate
    memory_estimate::String
end

"""
    create_shtnskit_config(; lmax::Int, mmax::Int=lmax, nlat::Int=lmax+2, 
                           nlon::Int=max(2*lmax+1, 4), optimize_decomp::Bool=true) -> SHTnsKitConfig

Create SHTnsKit configuration with MPI parallelization using PencilArrays.
This replaces the old SHTns configuration with a cleaner SHTnsKit-based approach.
"""
function create_shtnskit_config(; lmax::Int, mmax::Int=lmax, 
                               nlat::Int=max(lmax+2, i_Th), 
                               nlon::Int=max(2*lmax+1, 4, i_Ph),
                               optimize_decomp::Bool=true)
    
    # Create SHTnsKit configuration with Gauss-Legendre grid
    sht_config = SHTnsKit.create_gauss_config(lmax, nlat; 
                                            mmax=mmax, 
                                            nlon=nlon,
                                            norm=:orthonormal)
    
    # Enable optimized Legendre polynomial tables for better performance
    SHTnsKit.prepare_plm_tables!(sht_config)
    
    # Get MPI communicator
    comm = get_comm()
    nprocs = get_nprocs()
    
    # Create pencil decomposition for parallel theta-phi transforms
    pencils = create_pencil_decomposition_shtnskit(nlat, nlon, i_N, comm, optimize_decomp)
    
    # Create optimized transform plan for repeated operations
    plan = try
        SHTnsKit.SHTPlan(sht_config)
    catch e
        @warn "Could not create SHTPlan: $e"
        nothing
    end
    
    # Estimate memory usage
    field_count = estimate_field_count()
    memory_mb = estimate_memory_usage_shtnskit(nlat, nlon, lmax, field_count, Float64)
    memory_estimate = "$(round(memory_mb, digits=1)) MB"
    
    nlm = sht_config.nlm
    
    if get_rank() == 0
        print_shtnskit_config_summary(nlat, nlon, lmax, mmax, nlm, nprocs, memory_estimate)
    end
    
    return SHTnsKitConfig(sht_config, nlat, nlon, lmax, mmax, nlm, 
                         pencils, plan, memory_estimate)
end

"""
    create_pencil_decomposition_shtnskit(nlat, nlon, nr, comm, optimize)

Create PencilArrays decomposition optimized for theta-phi parallelization.
"""
function create_pencil_decomposition_shtnskit(nlat::Int, nlon::Int, nr::Int, 
                                             comm, optimize::Bool=true)
    nprocs = MPI.Comm_size(comm)
    
    # Determine optimal process topology for theta-phi parallelization
    if optimize && nprocs > 1
        proc_dims = optimize_process_topology_shtnskit(nprocs, nlat, nlon)
    else
        proc_dims = (nprocs, 1)
    end
    
    # Create PencilArrays topology
    topology = PencilArrays.Topology(comm, proc_dims)
    
    # Physical space pencils for theta-phi parallelization
    dims = (nlat, nlon, nr)
    pencil_theta = Pencil(topology, dims, (2, 3))  # Theta-contiguous
    pencil_phi = Pencil(topology, dims, (1, 3))    # Phi-contiguous  
    pencil_r = Pencil(topology, dims, (1, 2))      # Radial-contiguous
    
    # Spectral space pencil (for (l,m) modes)
    nlm = SHTnsKit.nlm_calc(maximum([nlat-1, nlon÷2]), 
                           min(maximum([nlat-1, nlon÷2]), 
                               minimum([nlat-1, nlon÷2])), 1)
    spec_dims = (nlm, 1, nr)
    pencil_spec = Pencil(topology, spec_dims, (2, 3))
    
    return (theta = pencil_theta, 
            phi = pencil_phi, 
            r = pencil_r,
            spec = pencil_spec)
end

"""
    optimize_process_topology_shtnskit(nprocs, nlat, nlon)

Optimize MPI process topology for theta-phi parallelization.
"""
function optimize_process_topology_shtnskit(nprocs::Int, nlat::Int, nlon::Int)
    # Find factorization that balances theta and phi parallelization
    best_dims = (nprocs, 1)
    best_score = Inf
    
    for p_theta in 1:nprocs
        if nprocs % p_theta == 0
            p_phi = nprocs ÷ p_theta
            
            # Check load balance
            theta_imbalance = abs(nlat % p_theta) / nlat
            phi_imbalance = abs(nlon % p_phi) / nlon
            
            # Prefer more balanced decomposition
            score = theta_imbalance + phi_imbalance
            
            if score < best_score
                best_score = score
                best_dims = (p_theta, p_phi)
            end
        end
    end
    
    return best_dims
end

"""
    estimate_memory_usage_shtnskit(nlat, nlon, lmax, field_count, T)

Estimate memory usage for SHTnsKit-based transforms.
"""
function estimate_memory_usage_shtnskit(nlat::Int, nlon::Int, lmax::Int, 
                                       field_count::Int, ::Type{T}) where T
    
    # Physical grid memory
    physical_memory = nlat * nlon * i_N * sizeof(T)
    
    # Spectral memory (approximate)
    nlm = (lmax + 1) * (lmax + 2) ÷ 2
    spectral_memory = nlm * i_N * sizeof(ComplexF64) * 2  # real + imag
    
    # Working arrays (SHTnsKit buffers)
    working_memory = max(physical_memory, spectral_memory) * 2
    
    # Total per field
    per_field_memory = physical_memory + spectral_memory + working_memory
    
    # Total for all fields
    total_memory = per_field_memory * field_count
    
    return total_memory / (1024^2)  # Convert to MB
end

"""
    print_shtnskit_config_summary(nlat, nlon, lmax, mmax, nlm, nprocs, memory_estimate)

Print configuration summary for SHTnsKit setup.
"""
function print_shtnskit_config_summary(nlat, nlon, lmax, mmax, nlm, nprocs, memory_estimate)
    println("\n╔═══════════════════════════════════════════════════════╗")
    println("║         SHTnsKit Configuration Summary                ║")
    println("╠═══════════════════════════════════════════════════════╣")
    println("║ Grid Configuration:                                   ║")
    println("║   Physical grid:    $(lpad(nlat,4)) × $(lpad(nlon,4)) × $(lpad(i_N,4))         ║")
    println("║   Spectral modes:   lmax=$(lpad(lmax,3)), mmax=$(lpad(mmax,3))              ║")
    println("║   Total modes:      $(lpad(nlm,5))                             ║")
    println("║                                                       ║")
    println("║ Parallel Configuration:                               ║")
    println("║   MPI Processes:    $(lpad(nprocs,4))                              ║")
    println("║   Theta-Phi Parallel: Enabled                        ║")
    println("║   Memory/process:   $(lpad(memory_estimate,10))                    ║")
    println("╚═══════════════════════════════════════════════════════╝")
end

# ============================================================================
# Core Transform Functions using SHTnsKit
# ============================================================================

"""
    shtnskit_spectral_to_physical!(spec::SHTnsSpectralField{T}, 
                                  phys::SHTnsPhysicalField{T}) where T

Transform from spectral to physical space using SHTnsKit with MPI parallelization.
This function parallelizes across theta and phi directions using PencilArrays.
"""
function shtnskit_spectral_to_physical!(spec::SHTnsSpectralField{T}, 
                                       phys::SHTnsPhysicalField{T}) where T
    config = spec.config
    sht_config = config.sht_config
    
    # Get local data ranges
    spec_real = parent(spec.data_real)
    spec_imag = parent(spec.data_imag) 
    phys_data = parent(phys.data)
    
    # Get local ranges for parallel decomposition
    r_range = range_local(config.pencils.r, 3)
    
    # Process each radial level with theta-phi parallelization
    @threads for r_idx in r_range
        local_r = r_idx - first(r_range) + 1
        
        if local_r <= size(spec_real, 3)
            # Extract spectral coefficients for this radial level
            coeffs = extract_spectral_coefficients(spec_real, spec_imag, local_r, config)
            
            # Perform SHTnsKit synthesis with parallelization
            if config.plan !== nothing
                # Use optimized plan for better performance
                phys_slice = zeros(T, config.nlat, config.nlon)
                SHTnsKit.synthesis!(config.plan, phys_slice, coeffs)
                
                # Store result with theta-phi parallelization
                store_physical_slice!(phys_data, phys_slice, local_r, config)
            else
                # Fallback to direct synthesis
                phys_slice = SHTnsKit.synthesis(sht_config, coeffs; real_output=true)
                store_physical_slice!(phys_data, phys_slice, local_r, config)
            end
        end
    end
    
    # Synchronize across MPI processes
    MPI.Barrier(get_comm())
end

"""
    shtnskit_physical_to_spectral!(phys::SHTnsPhysicalField{T}, 
                                  spec::SHTnsSpectralField{T}) where T

Transform from physical to spectral space using SHTnsKit with MPI parallelization.
"""
function shtnskit_physical_to_spectral!(phys::SHTnsPhysicalField{T}, 
                                       spec::SHTnsSpectralField{T}) where T
    config = spec.config
    sht_config = config.sht_config
    
    # Get local data
    phys_data = parent(phys.data)
    spec_real = parent(spec.data_real)
    spec_imag = parent(spec.data_imag)
    
    # Get local ranges
    r_range = range_local(config.pencils.r, 3)
    
    # Process each radial level with parallelization
    @threads for r_idx in r_range
        local_r = r_idx - first(r_range) + 1
        
        if local_r <= size(phys_data, 3)
            # Extract physical slice
            phys_slice = extract_physical_slice(phys_data, local_r, config)
            
            # Perform SHTnsKit analysis
            if config.plan !== nothing
                # Use optimized plan
                coeffs = zeros(ComplexF64, config.lmax+1, config.mmax+1)
                SHTnsKit.analysis!(config.plan, coeffs, phys_slice)
            else
                # Direct analysis
                coeffs = SHTnsKit.analysis(sht_config, phys_slice)
            end
            
            # Store spectral coefficients with parallelization
            store_spectral_coefficients!(spec_real, spec_imag, coeffs, local_r, config)
        end
    end
    
    # Synchronize across MPI processes
    MPI.Barrier(get_comm())
end

"""
    shtnskit_vector_synthesis!(tor_spec::SHTnsSpectralField{T}, 
                              pol_spec::SHTnsSpectralField{T},
                              vec_phys::SHTnsVectorField{T}) where T

Vector synthesis using SHTnsKit spheroidal-toroidal decomposition.
"""
function shtnskit_vector_synthesis!(tor_spec::SHTnsSpectralField{T}, 
                                   pol_spec::SHTnsSpectralField{T},
                                   vec_phys::SHTnsVectorField{T}) where T
    config = tor_spec.config
    sht_config = config.sht_config
    
    # Get local data
    tor_real = parent(tor_spec.data_real)
    tor_imag = parent(tor_spec.data_imag)
    pol_real = parent(pol_spec.data_real)
    pol_imag = parent(pol_spec.data_imag)
    
    v_theta = parent(vec_phys.θ_component.data)
    v_phi = parent(vec_phys.φ_component.data)
    
    # Get local ranges for parallelization
    r_range = range_local(config.pencils.r, 3)
    
    # Process each radial level with theta-phi parallelization
    @threads for r_idx in r_range
        local_r = r_idx - first(r_range) + 1
        
        if local_r <= size(tor_real, 3)
            # Extract toroidal and poloidal coefficients
            tor_coeffs = extract_spectral_coefficients(tor_real, tor_imag, local_r, config)
            pol_coeffs = extract_spectral_coefficients(pol_real, pol_imag, local_r, config)
            
            # Perform vector synthesis using SHTnsKit
            vt_field, vp_field = SHTnsKit.SHsphtor_to_spat(sht_config, pol_coeffs, tor_coeffs; 
                                                          real_output=true)
            
            # Store vector components with parallelization
            store_vector_components!(v_theta, v_phi, vt_field, vp_field, local_r, config)
        end
    end
    
    # Synchronize across processes
    MPI.Barrier(get_comm())
end

"""
    shtnskit_vector_analysis!(vec_phys::SHTnsVectorField{T},
                             tor_spec::SHTnsSpectralField{T}, 
                             pol_spec::SHTnsSpectralField{T}) where T

Vector analysis using SHTnsKit spheroidal-toroidal decomposition.
"""
function shtnskit_vector_analysis!(vec_phys::SHTnsVectorField{T},
                                  tor_spec::SHTnsSpectralField{T}, 
                                  pol_spec::SHTnsSpectralField{T}) where T
    config = tor_spec.config
    sht_config = config.sht_config
    
    # Get local data
    v_theta = parent(vec_phys.θ_component.data)
    v_phi = parent(vec_phys.φ_component.data)
    
    tor_real = parent(tor_spec.data_real)
    tor_imag = parent(tor_spec.data_imag)
    pol_real = parent(pol_spec.data_real)
    pol_imag = parent(pol_spec.data_imag)
    
    # Get local ranges
    r_range = range_local(config.pencils.r, 3)
    
    # Process each radial level with parallelization
    @threads for r_idx in r_range
        local_r = r_idx - first(r_range) + 1
        
        if local_r <= size(v_theta, 3)
            # Extract vector components
            vt_field = extract_physical_slice(v_theta, local_r, config)
            vp_field = extract_physical_slice(v_phi, local_r, config)
            
            # Perform vector analysis using SHTnsKit
            pol_coeffs, tor_coeffs = SHTnsKit.spat_to_SHsphtor(sht_config, vt_field, vp_field)
            
            # Store spectral coefficients
            store_spectral_coefficients!(pol_real, pol_imag, pol_coeffs, local_r, config)
            store_spectral_coefficients!(tor_real, tor_imag, tor_coeffs, local_r, config)
        end
    end
    
    # Synchronize across processes
    MPI.Barrier(get_comm())
end

# ============================================================================
# Helper Functions for Data Management
# ============================================================================

"""
    extract_spectral_coefficients(spec_real, spec_imag, local_r, config)

Extract spectral coefficients for a radial level with proper MPI communication.
"""
function extract_spectral_coefficients(spec_real, spec_imag, local_r, config)
    nlm = config.nlm
    lmax, mmax = config.lmax, config.mmax
    
    # Initialize coefficient matrix
    coeffs = zeros(ComplexF64, lmax+1, mmax+1)
    
    # Get local spectral data
    if local_r <= size(spec_real, 3)
        for lm_idx in 1:min(size(spec_real, 1), nlm)
            # Convert from packed format to (l,m) matrix format
            l, m = index_to_lm(lm_idx, lmax, mmax)
            if l <= lmax && m <= mmax
                real_part = spec_real[lm_idx, 1, local_r]
                imag_part = spec_imag[lm_idx, 1, local_r]
                coeffs[l+1, m+1] = complex(real_part, imag_part)
            end
        end
    end
    
    # Communicate across MPI processes to gather complete spectrum
    MPI.Allreduce!(coeffs, MPI.SUM, get_comm())
    
    return coeffs
end

"""
    store_spectral_coefficients!(spec_real, spec_imag, coeffs, local_r, config)

Store spectral coefficients with proper parallel decomposition.
"""
function store_spectral_coefficients!(spec_real, spec_imag, coeffs, local_r, config)
    lmax, mmax = config.lmax, config.mmax
    
    if local_r <= size(spec_real, 3)
        for lm_idx in 1:min(size(spec_real, 1), config.nlm)
            # Convert from (l,m) matrix to packed format
            l, m = index_to_lm(lm_idx, lmax, mmax)
            if l <= lmax && m <= mmax
                coeff = coeffs[l+1, m+1]
                spec_real[lm_idx, 1, local_r] = real(coeff)
                spec_imag[lm_idx, 1, local_r] = imag(coeff)
                
                # Ensure m=0 modes are real
                if m == 0
                    spec_imag[lm_idx, 1, local_r] = 0.0
                end
            end
        end
    end
end

"""
    extract_physical_slice(phys_data, local_r, config)

Extract a physical space slice for a given radial level.
"""
function extract_physical_slice(phys_data, local_r, config)
    nlat, nlon = config.nlat, config.nlon
    slice = zeros(eltype(phys_data), nlat, nlon)
    
    if local_r <= size(phys_data, 3)
        for i in 1:nlat, j in 1:nlon
            if i <= size(phys_data, 1) && j <= size(phys_data, 2)
                slice[i, j] = phys_data[i, j, local_r]
            end
        end
    end
    
    return slice
end

"""
    store_physical_slice!(phys_data, phys_slice, local_r, config)

Store a physical space slice with proper bounds checking.
"""
function store_physical_slice!(phys_data, phys_slice, local_r, config)
    if local_r <= size(phys_data, 3)
        for i in 1:size(phys_slice, 1), j in 1:size(phys_slice, 2)
            if i <= size(phys_data, 1) && j <= size(phys_data, 2)
                phys_data[i, j, local_r] = phys_slice[i, j]
            end
        end
    end
end

"""
    store_vector_components!(v_theta, v_phi, vt_field, vp_field, local_r, config)

Store vector components with proper parallelization.
"""
function store_vector_components!(v_theta, v_phi, vt_field, vp_field, local_r, config)
    if local_r <= size(v_theta, 3) && local_r <= size(v_phi, 3)
        for i in 1:size(vt_field, 1), j in 1:size(vt_field, 2)
            if i <= size(v_theta, 1) && j <= size(v_theta, 2) && 
               i <= size(v_phi, 1) && j <= size(v_phi, 2)
                v_theta[i, j, local_r] = vt_field[i, j]
                v_phi[i, j, local_r] = vp_field[i, j]
            end
        end
    end
end

"""
    index_to_lm(idx, lmax, mmax)

Convert linear index to (l,m) indices for spherical harmonics.
"""
function index_to_lm(idx::Int, lmax::Int, mmax::Int)
    # Simple conversion - can be optimized with lookup tables
    current_idx = 0
    for l in 0:lmax
        for m in 0:min(l, mmax)
            current_idx += 1
            if current_idx == idx
                return l, m
            end
        end
    end
    return 0, 0  # fallback
end

# ============================================================================
# Batch Processing for Better Performance
# ============================================================================

"""
    batch_shtnskit_transforms!(specs::Vector{SHTnsSpectralField{T}},
                              physs::Vector{SHTnsPhysicalField{T}}) where T

Batch process multiple spectral to physical transforms for better performance.
"""
function batch_shtnskit_transforms!(specs::Vector{SHTnsSpectralField{T}},
                                   physs::Vector{SHTnsPhysicalField{T}}) where T
    @assert length(specs) == length(physs)
    
    if isempty(specs)
        return
    end
    
    config = specs[1].config
    
    # Process in parallel batches for better cache utilization
    @threads for batch_idx in 1:length(specs)
        shtnskit_spectral_to_physical!(specs[batch_idx], physs[batch_idx])
    end
end

# ============================================================================
# Performance Monitoring
# ============================================================================

"""
    get_shtnskit_performance_stats()

Get performance statistics for SHTnsKit transforms.
"""
function get_shtnskit_performance_stats()
    # Return basic performance metrics
    return (
        library = "SHTnsKit",
        parallelization = "theta-phi MPI",
        optimization = "enabled"
    )
end

# ============================================================================
# Export functions
# ============================================================================

export SHTnsKitConfig, create_shtnskit_config
export shtnskit_spectral_to_physical!, shtnskit_physical_to_spectral!
export shtnskit_vector_synthesis!, shtnskit_vector_analysis!
export batch_shtnskit_transforms!
export get_shtnskit_performance_stats