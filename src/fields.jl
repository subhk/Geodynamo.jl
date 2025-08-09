# ============================================================================
# Variable Types with SHTns Integration
# ============================================================================
    
# Field types that work with PencilArrays
struct SHTnsSpectralField{T<:Number}
    config::SHTnsConfig
    nlm::Int
    data_real::PencilArray{T,3}
    data_imag::PencilArray{T,3}
    pencil::Pencil{3}  # Store pencil for local range info
end

# Physical field on SHTns grid
struct SHTnsPhysicalField{T<:Number}
    config::SHTnsConfig
    nlat::Int
    nlon::Int
    data::PencilArray{T,3}  # Single array, transpose as needed
    pencil::Pencil{3}       # Current pencil orientation
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

# Constructor functions compatible with PencilArrays
function create_shtns_spectral_field(::Type{T}, config::SHTnsConfig, 
                                    oc_domain::RadialDomain,
                                    pencil_spec::Pencil{3}) where T
    nlm = config.nlm
    
    # Create PencilArrays with the given pencil
    data_real = PencilArray{T}(undef, pencil_spec)
    data_imag = PencilArray{T}(undef, pencil_spec)
    
    # Initialize to zero
    fill!(parent(data_real), zero(T))
    fill!(parent(data_imag), zero(T))
    
    return SHTnsSpectralField{T}(config, nlm, 
                        data_real, data_imag, pencil_spec)
end


function create_shtns_physical_field(::Type{T}, config::SHTnsConfig,
                                    oc_domain::RadialDomain,
                                    pencil::Pencil{3}) where T
    nlat = config.nlat
    nlon = config.nlon
    
    # Create a single PencilArray
    data = PencilArray{T}(undef, pencil)
    fill!(parent(data), zero(T))
    
    return SHTnsPhysicalField{T}(config, nlat, nlon, data, pencil)
end


function create_shtns_vector_field(::Type{T}, config::SHTnsConfig,
                                    oc_domain::RadialDomain,
                                    pencils) where T
    pencil_θ, pencil_φ, pencil_r = pencils
    
    # Create each component with the r-pencil (contiguous in r)
    r_comp = create_shtns_physical_field(T, config, oc_domain, pencil_r)
    θ_comp = create_shtns_physical_field(T, config, oc_domain, pencil_r)
    φ_comp = create_shtns_physical_field(T, config, oc_domain, pencil_r)
    
    return SHTnsVectorField{T}(r_comp, θ_comp, φ_comp)
end


function create_radial_domain(nr::Int=i_N)
    N = nr
    
    r = zeros(N, 7)
    for n in 1:N
        r[n, 4] = 0.5 * (1.0 + cos(π * (N - n) / (N - 1)))
    end
    
    ri = d_rratio / (1.0 - d_rratio)
    r[:, 4] .+= ri
    
    for p in 1:7
        if p != 4
            power = p - 4
            r[:, p] = r[:, 4] .^ power
        end
    end
    
    dr_matrices         = [zeros(2*i_KL+1, N) for _ in 1:3]
    radial_laplacian    = zeros(2*i_KL+1, N)
    integration_weights = zeros(N)
    
    return RadialDomain(N, 1:N, r, dr_matrices, radial_laplacian, integration_weights)
end


# Helper functions for working with local portions of PencilArrays
function get_local_range(pencil::Pencil{3}, dim::Int)
    return range_local(pencil, dim)
end

# range_local function for getting local index ranges in pencil decomposition
function range_local(pencil::Pencil{3}, dim::Int)
    # Get the local range for the specified dimension
    local_shape = pencil.axes_in[dim]
    return local_shape
end

function get_local_indices(pencil::Pencil{3})
    return range_local(pencil)
end

# Access patterns for PencilArrays
function local_data_size(field::SHTnsSpectralField{T}) where T
    return size_local(field.pencil)
end

function local_data_size(field::SHTnsPhysicalField{T}) where T
    return size_local(field.pencil)
end

# Safe accessors that respect PencilArray's local data
function get_local_data(field::SHTnsSpectralField{T}) where T
    return (real=parent(field.data_real), imag=parent(field.data_imag))
end

function get_local_data(field::SHTnsPhysicalField{T}) where T
    return parent(field.data)
end

# export SHTnsSpectralField, SHTnsPhysicalField, SHTnsVectorField, SHTnsTorPolField
# export RadialDomain, create_shtns_spectral_field, create_shtns_physical_field
# export create_shtns_vector_field, create_radial_domain
# end