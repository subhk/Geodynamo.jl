# ============================================================================
# Variable Types with SHTns Integration
# ============================================================================

# module VariableTypes
    # using PencilArrays
    # using StaticArrays
    # using SHTnsSpheres
    # using ..Parameters
    # using ..SHTnsSetup
    # using ..PencilSetup
    
# Spherical harmonic field using SHTns
struct SHTnsSpectralField{T<:Number}
    config::SHTnsConfig
    nlm::Int
    data_real::PencilArray{T,3}   # Real coefficients (nlm, 1, r)
    data_imag::PencilArray{T,3}   # Imaginary coefficients (nlm, 1, r)
    local_radial_range::UnitRange{Int}
end

# Physical field on SHTns grid
struct SHTnsPhysicalField{T<:Number}
    config::SHTnsConfig
    nlat::Int
    nlon::Int
    data_θ::PencilArray{T,3}      # Data distributed along θ (nlat, nlon, r)
    data_φ::PencilArray{T,3}      # Data distributed along φ
    data_r::PencilArray{T,3}      # Data distributed along r
    local_radial_range::UnitRange{Int}
end

# Vector field with SHTns
struct SHTnsVectorField{T<:Number}
    r_component::SHTnsPhysicalField{T}
    θ_component::SHTnsPhysicalField{T}
    φ_component::SHTnsPhysicalField{T}
end

# Toroidal-Poloidal decomposition with SHTns
struct SHTnsTorPolField{T<:Number}
    toroidal::SHTnsSpectralField{T}
    poloidal::SHTnsSpectralField{T}
end

# Radial domain (unchanged)
struct RadialDomain
    N::Int
    local_range::UnitRange{Int}
    r::Matrix{Float64}
    dr_matrices::Vector{Matrix{Float64}}
    radial_laplacian::Matrix{Float64}
    integration_weights::Vector{Float64}
end

# Constructor functions
function create_shtns_spectral_field(::Type{T}, config::SHTnsConfig, 
                                    radial_domain::RadialDomain,
                                    pencil_spec::Pencil{3}) where T
    nlm = config.nlm
    dims = (nlm, 1, radial_domain.N)
    
    data_real = PencilArray{T}(undef, pencil_spec)
    data_imag = PencilArray{T}(undef, pencil_spec)
    
    fill!(data_real, zero(T))
    fill!(data_imag, zero(T))
    
    return SHTnsSpectralField{T}(config, nlm, data_real, data_imag, 
                                radial_domain.local_range)
end

function create_shtns_physical_field(::Type{T}, config::SHTnsConfig,
                                    radial_domain::RadialDomain,
                                    pencil_θ, pencil_φ, pencil_r) where T
    nlat = config.nlat
    nlon = config.nlon
    
    data_θ = PencilArray{T}(undef, pencil_θ)
    data_φ = PencilArray{T}(undef, pencil_φ)
    data_r = PencilArray{T}(undef, pencil_r)
    
    fill!(data_θ, zero(T))
    fill!(data_φ, zero(T))
    fill!(data_r, zero(T))
    
    return SHTnsPhysicalField{T}(config, nlat, nlon, data_θ, data_φ, data_r,
                                radial_domain.local_range)
end

function create_shtns_vector_field(::Type{T}, config::SHTnsConfig,
                                    radial_domain::RadialDomain,
                                    pencils) where T
    pencil_θ, pencil_φ, pencil_r = pencils
    
    r_comp = create_shtns_physical_field(T, config, radial_domain, pencil_θ, pencil_φ, pencil_r)
    θ_comp = create_shtns_physical_field(T, config, radial_domain, pencil_θ, pencil_φ, pencil_r)
    φ_comp = create_shtns_physical_field(T, config, radial_domain, pencil_θ, pencil_φ, pencil_r)
    
    return SHTnsVectorField{T}(r_comp, θ_comp, φ_comp)
end

function create_radial_domain(pencil_r)
    # Get local radial range for this process
    local_range = pencil_r.axes[3]
    N = i_N
    
    # Initialize radial points (Chebyshev-Gauss-Lobatto)
    r = zeros(N, 7)  # r^p for p=-3:3
    for n in 1:N
        r[n, 4] = 0.5 * (1.0 + cos(π * (N - n) / (N - 1)))  # r^1
    end
    
    # Map to spherical shell
    ri = d_rratio / (1.0 - d_rratio)
    r[:, 4] .+= ri
    
    # Compute other powers
    for p in 1:7
        if p != 4
            power = p - 4
            r[:, p] = r[:, 4] .^ power
        end
    end
    
    # Initialize matrices (simplified - full computation needed)
    dr_matrices = [zeros(2*i_KL+1, N) for _ in 1:3]
    radial_laplacian = zeros(2*i_KL+1, N)
    integration_weights = zeros(N)
    
    return RadialDomain(N, local_range, r, dr_matrices, radial_laplacian, integration_weights)
end

# export SHTnsSpectralField, SHTnsPhysicalField, SHTnsVectorField, SHTnsTorPolField
# export RadialDomain, create_shtns_spectral_field, create_shtns_physical_field
# export create_shtns_vector_field, create_radial_domain
# end