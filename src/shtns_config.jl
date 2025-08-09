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


"""
    create_pencil_topology_shtns(config, optimize=true)
    
Create pencil topology specifically for SHTns configuration.
"""
function create_pencil_topology_shtns(config, optimize::Bool=true)
    comm = get_comm()
    rank = get_rank()
    nprocs = get_nprocs()
    
    # Grid dimensions
    nlat = config.nlat
    nlon = config.nlon
    nr = i_N
    dims = (nlat, nlon, nr)
    
    # Determine optimal process topology
    if optimize && nprocs > 1
        proc_dims = optimize_process_topology(nprocs, dims)
    else
        proc_dims = (nprocs, 1)
    end
    
    # Create PencilArrays topology
    topology = PencilArrays.Topology(comm, proc_dims)
    
    # Create specialized pencils for SHTns operations
    pencils = create_shtns_pencils(topology, dims, config)
    
    # Analyze load balance
    if rank == 0 && nprocs > 1
        println("\nLoad Balance Analysis:")
        for (name, pencil) in pairs(pencils)
            if name != :plans  # Skip if plans are included
                analyze_pencil_balance(pencil, name)
            end
        end
    end
    
    return pencils
end


"""
    create_shtns_pencils(topology, dims, config)
    
Create pencils optimized for SHTns operations.
"""
function create_shtns_pencils(topology::PencilArrays.Topology, 
                            dims::Tuple{Int,Int,Int}, 
                            config)
    nlat, nlon, nr = dims
    
    # Physical space pencils
    pencil_θ = Pencil(topology, dims, (2, 3))  # Contiguous in θ
    pencil_φ = Pencil(topology, dims, (1, 3))  # Contiguous in φ  
    pencil_r = Pencil(topology, dims, (1, 2))  # Contiguous in r
    
    # Spectral space pencil
    spec_dims = (config.nlm, 1, nr)
    pencil_spec = Pencil(topology, spec_dims, (2,))
    
    # Hybrid pencil for SHTns transforms
    # This is optimized for the synthesis/analysis operations
    hybrid_dims = (max(nlat, config.nlm), nlon, nr)
    pencil_hybrid = Pencil(topology, hybrid_dims, (2, 3))
    
    return (θ = pencil_θ, 
            φ = pencil_φ, 
            r = pencil_r,
            spec = pencil_spec,
            hybrid = pencil_hybrid)
end


"""
    analyze_pencil_balance(pencil, name)
    
Analyze load balance for a specific pencil.
"""
function analyze_pencil_balance(pencil::Pencil, name::Symbol)
    comm = get_comm()
    rank = get_rank()
    
    local_size = prod(size_local(pencil))
    all_sizes = MPI.Gather(local_size, comm; root=0)
    
    if rank == 0 && !isempty(all_sizes)
        min_size = minimum(all_sizes)
        max_size = maximum(all_sizes)
        imbalance = (max_size - min_size) / max_size * 100
        
        status = imbalance < 5 ? "✓" : imbalance < 15 ? "○" : "✗"
        println("  $name pencil: $status $(round(imbalance, digits=1))% imbalance")
    end
end


"""
    estimate_field_count()
    
Estimate number of fields for memory calculation.
"""
function estimate_field_count()
    # Count fields based on problem configuration
    field_count = 0
    
    # Velocity: 3 vector components + toroidal/poloidal
    field_count += 5
    
    # Magnetic field (if enabled)
    if i_B == 1
        field_count += 5
    end
    
    # Temperature
    field_count += 2
    
    # Work arrays (estimate)
    field_count += 10
    
    return field_count
end



"""
    print_shtns_config_summary(args...)
    
Print configuration summary.
"""
function print_shtns_config_summary(nlat, nlon, lmax, mmax, nlm, 
                                   nprocs, mem_str, optimized)
    println("\n╔═══════════════════════════════════════════════════════╗")
    println("║         SHTns Configuration Summary                    ║")
    println("╠═══════════════════════════════════════════════════════╣")
    println("║ Grid Configuration:                                    ║")
    println("║   Physical grid:    $(lpad(nlat,4)) × $(lpad(nlon,4)) × $(lpad(i_N,4))          ║")
    println("║   Spectral modes:   lmax=$(lpad(lmax,3)), mmax=$(lpad(mmax,3))               ║")
    println("║   Total modes:      $(lpad(nlm,5))                              ║")
    println("║                                                        ║")
    println("║ Parallel Configuration:                                ║")
    println("║   MPI Processes:    $(lpad(nprocs,4))                               ║")
    println("║   Decomposition:    $(optimized ? "Optimized" : "Default  ")                          ║")
    println("║   Memory/process:   $(lpad(mem_str,10))                     ║")
    println("╚═══════════════════════════════════════════════════════╝")
end


"""
    get_mode_index(config::SHTnsConfig, l::Int, m::Int)
    
Get linear index for (l,m) mode.
"""
function get_mode_index(config::SHTnsConfig, l::Int, m::Int)
    return get(config.lm_index, (l, m), 0)
end

"""
    is_mode_local(config::SHTnsConfig, l::Int, m::Int)
    
Check if (l,m) mode is on local process.
"""
function is_mode_local(config::SHTnsConfig, l::Int, m::Int)
    idx = get_mode_index(config, l, m)
    if idx == 0
        return false
    end
    
    lm_range = range_local(config.pencils.spec, 1)
    return idx in lm_range
end


"""
    get_local_modes(config::SHTnsConfig)
    
Get list of (l,m) modes on local process.
"""
function get_local_modes(config::SHTnsConfig)
    lm_range = range_local(config.pencils.spec, 1)
    local_modes = Tuple{Int,Int}[]
    
    for idx in lm_range
        if idx <= config.nlm
            push!(local_modes, (config.l_values[idx], config.m_values[idx]))
        end
    end
    
    return local_modes
end



"""
    validate_config(config::SHTnsConfig)
    
Validate SHTns configuration for consistency.
"""
function validate_config(config::SHTnsConfig)
    errors = String[]
    
    # Check grid sizes
    if config.nlat < config.lmax + 1
        push!(errors, "nlat ($(config.nlat)) should be ≥ lmax+1 ($(config.lmax+1))")
    end
    
    if config.nlon < 2 * config.mmax + 1
        push!(errors, "nlon ($(config.nlon)) should be ≥ 2*mmax+1 ($(2*config.mmax+1))")
    end
    
    # Check mode counts
    expected_nlm = sum(l -> min(l, config.mmax) + 1, 0:config.lmax)
    if config.nlm != expected_nlm
        push!(errors, "nlm mismatch: got $(config.nlm), expected $expected_nlm")
    end
    
    # Check pencil compatibility
    for (name, pencil) in pairs(config.pencils)
        if name != :plans && name != :hybrid
            global_size = size_global(pencil)
            if name == :spec
                if global_size[1] != config.nlm
                    push!(errors, "Spectral pencil size mismatch")
                end
            else
                if global_size[1] != config.nlat || global_size[2] != config.nlon
                    push!(errors, "$name pencil size mismatch")
                end
            end
        end
    end
    
    if !isempty(errors)
        error("Configuration validation failed:\n" * join(errors, "\n"))
    end
    
    return true
end


# Export functions
# export SHTnsConfig, create_shtns_config
# export get_mode_index, is_mode_local, get_local_modes
# export validate_config
