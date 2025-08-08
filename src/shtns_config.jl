# =======================================================
# SHTns Configuration with Pencil Integration
# =======================================================
#
# SHTns configuration structure
struct SHTnsConfig
    sht::SHTnsSphere
    nlat::Int
    nlon::Int
    lmax::Int
    mmax::Int
    nlm::Int                                # Number of (l,m) modes
    
    l_values::Vector{Int}                   # l values for each mode
    m_values::Vector{Int}                   # m values for each mode

    lm_index::Dict{Tuple{Int,Int}, Int}     # (l,m) -> index mapping
    theta_grid::Vector{Float64}             # Colatitude grid points
    phi_grid::Vector{Float64}               # Azimuth grid points
    
    gauss_weights::Vector{Float64}          # Gaussian quadrature weights
    
    pencils::NamedTuple                     # Pencil decomposition
    transpose_plans::Dict{Symbol, Any}      # Transpose plans
    memory_estimate::String                 # Estimated memory usage
end


"""
    create_shtns_config(; optimize_decomp=true, enable_timing=false)
    
Create SHTns configuration with optimized parallel decomposition.
"""
function create_shtns_config(; optimize_decomp::Bool=true, 
                            enable_timing::Bool=false)
    # Initialize MPI if needed
    comm = get_comm()
    rank = get_rank()
    nprocs = get_nprocs()
    
    # Adjust grid sizes for SHTns compatibility
    nlat = compute_optimal_nlat(i_Th, i_L)
    nlon = compute_optimal_nlon(i_Ph, i_M)
    lmax = i_L
    mmax = min(i_M, lmax)
    
    # Initialize SHTns sphere
    sht = SHTnsSphere(lmax, mmax, 
                    grid_type = SHTnsSpheres.gaussian,
                    nlat = nlat,
                    nlon = nlon)
    
    # Get grid information
    theta_grid    = get_theta_array(sht)
    phi_grid      = get_phi_array(sht) 
    gauss_weights = get_weights(sht)
    
    # Compute (l,m) mode information with index mapping
    nlm = get_nlm(sht)
    l_values = zeros(Int, nlm)
    m_values = zeros(Int, nlm)
    lm_index = Dict{Tuple{Int,Int}, Int}()
    
    idx = 1
    for l in 0:lmax
        for m in 0:min(l, mmax)
            if idx <= nlm
                l_values[idx] = l
                m_values[idx] = m
                lm_index[(l, m)] = idx
                idx += 1
            end
        end
    end
    
    # Create optimized pencil decomposition
    if enable_timing
        ENABLE_TIMING[] = true
    end
    
    # Create temporary config for pencil creation
    temp_config = (nlat=nlat, nlon=nlon, nlm=nlm)
    pencils = create_pencil_topology_shtns(temp_config, optimize=optimize_decomp)
    
    # Create transpose plans
    transpose_plans = create_transpose_plans(pencils)
    
    # Estimate memory usage
    field_count = estimate_field_count()
    bytes, mem_str = estimate_memory_usage(pencils, field_count, Float64)
    
    # Print configuration summary
    if rank == 0
        print_shtns_config_summary(nlat, nlon, lmax, mmax, nlm, 
                                  nprocs, mem_str, optimize_decomp)
    end
    
    return SHTnsConfig(sht, nlat, nlon, lmax, mmax, nlm,
                      l_values, m_values, lm_index,
                      theta_grid, phi_grid, gauss_weights,
                      pencils, transpose_plans, mem_str)
end


"""
    compute_optimal_nlat(target_nlat::Int, lmax::Int)
    
Compute optimal latitude grid size for SHTns.
"""
function compute_optimal_nlat(target_nlat::Int, lmax::Int)
    # For Gaussian grid: nlat should be roughly (lmax+1)
    # Also ensure even number for FFT efficiency
    nlat = max(target_nlat, lmax + 1)
    
    # Ensure divisibility for good parallel decomposition
    nprocs = get_nprocs()
    if nprocs > 1
        # Round up to nearest number divisible by small factors
        while nlat % 2 != 0 || (nprocs <= 8 && nlat % min(nprocs, 4) != 0)
            nlat += 1
        end
    else
        # Just ensure even for FFT
        if nlat % 2 != 0
            nlat += 1
        end
    end
    
    return nlat
end


"""
    compute_optimal_nlon(target_nlon::Int, mmax::Int)
    
Compute optimal longitude grid size for SHTns.
"""
function compute_optimal_nlon(target_nlon::Int, mmax::Int)
    # For longitude: nlon should be at least 2*mmax + 1
    nlon = max(target_nlon, 2 * mmax + 1)
    
    # Round up to power of 2 or highly composite number for FFT efficiency
    nlon = next_fft_size(nlon)
    
    return nlon
end



"""
    next_fft_size(n::Int)
    
Find next size optimal for FFT (powers of 2, 3, 5, 7).
"""
function next_fft_size(n::Int)
    # Find next number that is a product of small primes
    while true
        m = n
        for p in [2, 3, 5, 7]
            while m % p == 0
                m ÷= p
            end
        end
        if m == 1
            return n
        end
        n += 1
    end
end



# Parallel decomposition with SHTns
function create_parallel_shtns_config()
    comm = PencilSetup.get_comm()
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)
    
    config = create_shtns_config()
    
    if rank == 0
        println("SHTns Configuration:")
        println("  Grid: $(config.nlat) × $(config.nlon)")
        println("  lmax: $(config.lmax), mmax: $(config.mmax)")
        println("  Number of modes: $(config.nlm)")
        println("  Processes: $nprocs")
    end
    
    return config
end

#export SHTnsConfig, create_shtns_config, create_parallel_shtns_config