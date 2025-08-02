using SHTnsSpheres
using MPI
using ..Parameters
    
# SHTns configuration structure
struct SHTnsConfig
    sht::SHTnsSphere
    nlat::Int
    nlon::Int
    lmax::Int
    mmax::Int
    nlm::Int                    # Number of (l,m) modes
    l_values::Vector{Int}       # l values for each mode
    m_values::Vector{Int}       # m values for each mode
    theta_grid::Vector{Float64} # Colatitude grid points
    phi_grid::Vector{Float64}   # Azimuth grid points
    gauss_weights::Vector{Float64} # Gaussian quadrature weights
end

function create_shtns_config()
    # Initialize SHTns with optimized settings
    # SHTns requires specific grid sizes for optimal performance
    
    # Adjust grid sizes to be compatible with SHTns
    # For Gaussian grid: nlat should be roughly (lmax+1)
    # For longitude: nlon should be at least 2*mmax
    nlat = max(i_Th, i_L + 1)
    nlon = max(i_Ph, 2 * i_M)
    
    # Ensure even number for FFT efficiency
    if nlat % 2 != 0
        nlat += 1
    end
    if nlon % 2 != 0
        nlon += 1
    end
    
    lmax = i_L
    mmax = min(i_M, lmax)  # mmax cannot exceed lmax
    
    # Initialize SHTns sphere
    # Use Gaussian grid for optimal accuracy
    sht = SHTnsSphere(lmax, mmax, 
                        grid_type = SHTnsSpheres.gaussian,
                        nlat = nlat,
                        nlon = nlon)
    
    # Get grid information
    theta_grid = get_theta_array(sht)
    phi_grid = get_phi_array(sht) 
    gauss_weights = get_weights(sht)
    
    # Compute (l,m) mode information
    nlm = get_nlm(sht)
    l_values = zeros(Int, nlm)
    m_values = zeros(Int, nlm)
    
    idx = 1
    for l in 0:lmax
        for m in 0:min(l, mmax)
            if idx <= nlm
                l_values[idx] = l
                m_values[idx] = m
                idx += 1
            end
        end
    end
    
    return SHTnsConfig(sht, nlat, nlon, lmax, mmax, nlm,
                        l_values, m_values, theta_grid, phi_grid, gauss_weights)

end

# Parallel decomposition with SHTns
function create_parallel_shtns_config(comm::MPI.Comm)
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)
    
    # Create base configuration
    config = create_shtns_config()
    
    if rank == 0
        println("SHTns Configuration:")
        println("  Grid: $(config.nlat) × $(config.nlon)")
        println("  lmax: $(config.lmax), mmax: $(config.mmax)")
        println("  Number of modes: $(config.nlm)")
        println("  Processes: $nprocs")
    end
    
    # SHTns can be used with MPI domain decomposition
    # Each process handles different radial levels and/or spectral modes
    
    return config
end

export SHTnsConfig, create_shtns_config, create_parallel_shtns_config