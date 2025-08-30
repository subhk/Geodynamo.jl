"""
Compatibility shims to ease migration from legacy SHTns-based API to SHTnsKit.
These wrappers map old types and function names to the new SHTnsKit versions
implemented in `shtnskit_transforms.jl`.
"""

# Type alias so existing code referring to `SHTnsConfig` keeps working
const SHTnsConfig = SHTnsKitConfig

# Minimal, no-op transform manager placeholder
struct SHTnsTransformManager{T} end

# Stub factory returning a placeholder (legacy code stores it but doesn't need it)
get_transform_manager(::Type{T}, ::SHTnsKitConfig) where {T} = SHTnsTransformManager{T}()

# Map legacy transform function names to SHTnsKit implementations
shtns_spectral_to_physical!(spec::SHTnsSpectralField{T}, phys::SHTnsPhysicalField{T}) where {T} =
    shtnskit_spectral_to_physical!(spec, phys)

shtns_physical_to_spectral!(phys::SHTnsPhysicalField{T}, spec::SHTnsSpectralField{T}) where {T} =
    shtnskit_physical_to_spectral!(phys, spec)

shtns_vector_synthesis!(tor::SHTnsSpectralField{T}, pol::SHTnsSpectralField{T}, vec::SHTnsVectorField{T}) where {T} =
    shtnskit_vector_synthesis!(tor, pol, vec)

shtns_vector_analysis!(vec::SHTnsVectorField{T}, tor::SHTnsSpectralField{T}, pol::SHTnsSpectralField{T}) where {T} =
    shtnskit_vector_analysis!(vec, tor, pol)

# Backward-compatible constructor name
create_shtns_config(; kwargs...) = create_shtnskit_config(; kwargs...)

