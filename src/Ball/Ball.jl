module GeodynamoBall

"""
Ball-specific convenience API.
Provides a radial domain and field constructors appropriate for a solid
sphere (inner radius = 0). Transforms reuse the core SHTnsKit machinery.
"""

using ..Geodynamo
using LinearAlgebra

# Re-export the core config type for clarity in ball context
const BallConfig = Geodynamo.SHTnsKitConfig

export BallConfig
export create_ball_pencils
export create_ball_radial_domain
export create_ball_spectral_field, create_ball_physical_field, create_ball_vector_field
export create_ball_velocity_fields, create_ball_temperature_field
export create_ball_composition_field, create_ball_magnetic_fields
export create_ball_hybrid_temperature_boundaries, create_ball_hybrid_composition_boundaries

"""
    create_ball_radial_domain(nr=i_N) -> RadialDomain

Create a radial domain for a solid sphere (inner radius = 0).
Uses a cosine-stretched grid similar to the shell for compatibility,
but sets the inner radius to zero and adjusts the coordinate columns
to match expectations of downstream operators.
"""
function create_ball_radial_domain(nr::Int = Geodynamo.i_N)
    N = nr
    # r[:,4] holds the base radius coordinate in existing code
    r = zeros(Float64, N, 7)
    # Cosine clustering towards r=0 and r=1 like Chebyshev nodes mapped to [0,1]
    for n in 1:N
        # x in [-1,1]
        x = cos(pi * (N - n) / (N - 1))
        # map to [0,1]
        r[n, 4] = 0.5 * (1.0 + x)
    end
    # Fill powers of r in other columns for compatibility
    for p in 1:7
        if p != 4
            power = p - 4
            r[:, p] = r[:, 4] .^ power
        end
    end

    dr_matrices         = [zeros(2*Geodynamo.i_KL+1, N) for _ in 1:3]
    radial_laplacian    = zeros(2*Geodynamo.i_KL+1, N)
    integration_weights = zeros(Float64, N)

    return Geodynamo.RadialDomain(N, 1:N, r, dr_matrices, radial_laplacian, integration_weights)
end

"""
    create_ball_spectral_field(T, cfg::BallConfig, domain::Geodynamo.RadialDomain, pencil)
"""
create_ball_spectral_field(::Type{T}, cfg::BallConfig, domain::Geodynamo.RadialDomain, pencil) where {T} =
    Geodynamo.create_shtns_spectral_field(T, cfg, domain, pencil)

"""
    create_ball_physical_field(T, cfg::BallConfig, domain::Geodynamo.RadialDomain, pencil)
"""
create_ball_physical_field(::Type{T}, cfg::BallConfig, domain::Geodynamo.RadialDomain, pencil) where {T} =
    Geodynamo.create_shtns_physical_field(T, cfg, domain, pencil)

"""
    create_ball_vector_field(T, cfg::BallConfig, domain::Geodynamo.RadialDomain, pencils)
"""
create_ball_vector_field(::Type{T}, cfg::BallConfig, domain::Geodynamo.RadialDomain, pencils) where {T} =
    Geodynamo.create_shtns_vector_field(T, cfg, domain, pencils)

"""
    create_ball_pencils(cfg::BallConfig; optimize=true)
"""
create_ball_pencils(cfg::BallConfig; optimize::Bool=true) = Geodynamo.create_pencil_topology(cfg; optimize)

"""
    create_ball_velocity_fields(T, cfg::BallConfig; nr=Geodynamo.i_N)
"""
function create_ball_velocity_fields(::Type{T}, cfg::BallConfig; nr::Int=Geodynamo.i_N) where {T}
    domain = create_ball_radial_domain(nr)
    pencils = create_ball_pencils(cfg)
    return Geodynamo.create_shtns_velocity_fields(T, cfg, domain, pencils, pencils.spec)
end

"""
    create_ball_temperature_field(T, cfg::BallConfig; nr=Geodynamo.i_N)
"""
function create_ball_temperature_field(::Type{T}, cfg::BallConfig; nr::Int=Geodynamo.i_N) where {T}
    domain = create_ball_radial_domain(nr)
    return Geodynamo.create_shtns_temperature_field(T, cfg, domain)
end

"""
    create_ball_composition_field(T, cfg::BallConfig; nr=Geodynamo.i_N)
"""
function create_ball_composition_field(::Type{T}, cfg::BallConfig; nr::Int=Geodynamo.i_N) where {T}
    domain = create_ball_radial_domain(nr)
    return Geodynamo.create_shtns_composition_field(T, cfg, domain)
end

"""
    create_ball_magnetic_fields(T, cfg::BallConfig; nr=Geodynamo.i_N)

Create magnetic fields for a solid sphere. Since a "core" split is not
used in a ball, we pass the same domain for both oc and ic to reuse the
core implementation.
"""
function create_ball_magnetic_fields(::Type{T}, cfg::BallConfig; nr::Int=Geodynamo.i_N) where {T}
    domain = create_ball_radial_domain(nr)
    pencils = create_ball_pencils(cfg)
    return Geodynamo.create_shtns_magnetic_fields(T, cfg, domain, domain, pencils, pencils.spec)
end

"""
    create_ball_hybrid_temperature_boundaries(inner_spec, outer_spec, cfg::BallConfig; precision=Float64)
"""
create_ball_hybrid_temperature_boundaries(inner_spec, outer_spec, cfg::BallConfig; precision::Type{T}=Float64) where {T} =
    Geodynamo.create_hybrid_temperature_boundaries(inner_spec, outer_spec, cfg; precision)

"""
    create_ball_hybrid_composition_boundaries(inner_spec, outer_spec, cfg::BallConfig; precision=Float64)
"""
create_ball_hybrid_composition_boundaries(inner_spec, outer_spec, cfg::BallConfig; precision::Type{T}=Float64) where {T} =
    Geodynamo.create_hybrid_composition_boundaries(inner_spec, outer_spec, cfg; precision)

end # module
