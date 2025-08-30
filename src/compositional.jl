# ============================================================================
# Compositional field components with full spectral optimization
# ============================================================================

using MPI
using PencilArrays
using SHTnsKit
using LinearAlgebra
using SparseArrays

struct SHTnsCompositionField{T}
    # Physical space composition
    composition::SHTnsPhysicalField{T}
    gradient::SHTnsVectorField{T}
    
    # Spectral representation
    spectral::SHTnsSpectralField{T}
    
    # Nonlinear terms (advection)
    nonlinear::SHTnsSpectralField{T}
    
    # Work arrays for efficient computation
    work_spectral::SHTnsSpectralField{T}
    work_physical::SHTnsPhysicalField{T}
    advection_physical::SHTnsPhysicalField{T}
    
    # Gradient spectral components for efficiency
    grad_theta_spec::SHTnsSpectralField{T}
    grad_phi_spec::SHTnsSpectralField{T}
    grad_r_spec::SHTnsSpectralField{T}
    
    # Sources and boundary conditions
    internal_sources::Vector{T}        # Radial profile of compositional sources
    boundary_values::Matrix{T}         # [2, nlm] for ICB and CMB
    bc_type_inner::Vector{Int}         # BC type for each mode at inner
    bc_type_outer::Vector{Int}         # BC type for each mode at outer
    
    # Pre-computed coefficients
    l_factors::Vector{Float64}         # l(l+1) values for diffusion
    config::SHTnsKitConfig             # SHTnsKit configuration
    
    # Radial derivative matrices
    dr_matrix::BandedMatrix{T}        # First derivative d/dr
    d2r_matrix::BandedMatrix{T}       # Second derivative d²/dr²
    
    # Spectral derivative operators (pre-computed)
    theta_derivative_matrix::Matrix{T}     # θ derivative coupling
    theta_recurrence_coeffs::Matrix{T}     # Recurrence relations
    
    # Performance tracking
    computation_time::Ref{Float64}
    transform_time::Ref{Float64}
end

function create_shtns_composition_field(::Type{T}, config::SHTnsKitConfig, 
                                        oc_domain::RadialDomain) where T
    # Use config's pencils directly
    pencils = config.pencils
    
    # Composition field in r-pencil for efficient radial operations
    composition = create_shtns_physical_field(T, config, oc_domain, pencils.r)
    
    # Gradient components
    gradient = create_shtns_vector_field(T, config, oc_domain, 
                                        (pencils.θ, pencils.φ, pencils.r))
    
    # Spectral representation in spectral pencil for efficient transforms
    spectral = create_shtns_spectral_field(T, config, oc_domain, pencils.spec)
    nonlinear = create_shtns_spectral_field(T, config, oc_domain, pencils.spec)
    
    # Work arrays
    work_spectral      = create_shtns_spectral_field(T, config, oc_domain, pencils.spec)
    work_physical      = create_shtns_physical_field(T, config, oc_domain, pencils.r)
    advection_physical = create_shtns_physical_field(T, config, oc_domain, pencils.r)
    
    # Gradient spectral components
    grad_theta_spec = create_shtns_spectral_field(T, config, oc_domain, pencils.spec)
    grad_phi_spec   = create_shtns_spectral_field(T, config, oc_domain, pencils.spec)
    grad_r_spec     = create_shtns_spectral_field(T, config, oc_domain, pencils.spec)
    
    # Sources and boundary conditions
    internal_sources = zeros(T, oc_domain.N)
    boundary_values  = zeros(T, 2, config.nlm)
    
    # Default boundary conditions (1 = fixed value, 2 = flux)
    # For composition: typically no-flux at both boundaries
    bc_type_inner = fill(2, config.nlm)  # No-flux at inner boundary
    bc_type_outer = fill(2, config.nlm)  # No-flux at outer boundary
    
    # Pre-compute l(l+1) factors for diffusion operator
    l_factors = zeros(Float64, config.nlm)
    for lm_idx in 1:config.nlm
        l = config.l_values[lm_idx]
        l_factors[lm_idx] = Float64(l * (l + 1))
    end
    
    # Transform manager removed in SHTnsKit migration
    
    # Create radial derivative matrices
    dr_matrix  = create_derivative_matrix(1, oc_domain)
    d2r_matrix = create_derivative_matrix(2, oc_domain)
    
    # Pre-compute spectral derivative operators
    theta_derivative_matrix = build_theta_derivative_matrix(T, config)
    theta_recurrence_coeffs = compute_theta_recurrence_coefficients(T, config)
    
    return SHTnsCompositionField{T}(
        composition, gradient, spectral, nonlinear,
        work_spectral, work_physical, advection_physical,
        grad_theta_spec, grad_phi_spec, grad_r_spec,
        internal_sources, boundary_values,
        bc_type_inner, bc_type_outer,
        l_factors, config,
        dr_matrix, d2r_matrix,
        theta_derivative_matrix, theta_recurrence_coeffs,
        Ref{Float64}(0.0), Ref{Float64}(0.0)
    )
end

function compute_composition_nonlinear!(comp_field::SHTnsCompositionField{T}, 
                                        vel_fields, oc_domain::RadialDomain) where T
    t_start = ENABLE_TIMING[] ? MPI.Wtime() : 0.0
    
    # Zero work arrays
    zero_composition_work_arrays!(comp_field)
    
    # Step 1: Transform composition to physical space for advection
    t_transform = MPI.Wtime()
    shtnskit_spectral_to_physical!(comp_field.spectral, comp_field.composition)
    comp_field.transform_time[] += MPI.Wtime() - t_transform
    
    # Step 2: Compute gradient in physical space if needed for diffusion
    compute_composition_gradient_local!(comp_field)
    
    # Step 3: Compute advection term -u·∇C in physical space (local operation)
    if vel_fields !== nothing
        compute_composition_advection_local!(comp_field, vel_fields)
    end
    
    # Step 4: Add internal compositional sources (local operation)
    add_compositional_sources_local!(comp_field, oc_domain)
    
    # Step 5: Transform advection + sources back to spectral space
    t_transform = MPI.Wtime()
    shtnskit_physical_to_spectral!(comp_field.advection_physical, comp_field.nonlinear)
    comp_field.transform_time[] += MPI.Wtime() - t_transform
    
    # Step 6: Apply boundary conditions in spectral space
    apply_composition_boundary_conditions_spectral!(comp_field, oc_domain)
    
    if ENABLE_TIMING[]
        comp_field.computation_time[] += MPI.Wtime() - t_start
    end
end

function zero_composition_work_arrays!(comp_field::SHTnsCompositionField{T}) where T
    fill!(parent(comp_field.work_spectral.data_real), zero(T))
    fill!(parent(comp_field.work_spectral.data_imag), zero(T))
    fill!(parent(comp_field.work_physical.data), zero(T))
    fill!(parent(comp_field.advection_physical.data), zero(T))
    fill!(parent(comp_field.nonlinear.data_real), zero(T))
    fill!(parent(comp_field.nonlinear.data_imag), zero(T))
end

function compute_composition_gradient_local!(comp_field::SHTnsCompositionField{T}) where T
    # Compute ∇C in physical space for use in diffusion and analysis
    # This is typically done in spectral space for efficiency, but can be
    # computed locally if needed for specific applications
    
    # For now, we'll compute the gradient in spectral space when needed
    # This function serves as a placeholder for local gradient computation
    return
end

function compute_composition_advection_local!(comp_field::SHTnsCompositionField{T}, 
                                              vel_fields) where T
    # Compute u·∇C advection term in physical space
    
    # Get local data arrays
    comp_data = parent(comp_field.composition.data)
    adv_data = parent(comp_field.advection_physical.data)
    
    # Get velocity data (assuming it's already in physical space)
    u_r_data = parent(vel_fields.velocity.r_component.data)
    u_θ_data = parent(vel_fields.velocity.θ_component.data)
    u_φ_data = parent(vel_fields.velocity.φ_component.data)
    
    # Zero the advection array
    fill!(adv_data, zero(T))
    
    # Compute advection: -u·∇C
    # This is a simplified implementation - full implementation would include
    # proper gradient computation in spherical coordinates
    
    nlat, nlon, nr = size(comp_data)
    
    for r_idx in 1:nr
        for φ_idx in 1:nlon
            for θ_idx in 1:nlat
                if (θ_idx <= size(u_r_data, 1) && φ_idx <= size(u_r_data, 2) && 
                    r_idx <= size(u_r_data, 3))
                    
                    # Get velocity components
                    u_r = u_r_data[θ_idx, φ_idx, r_idx]
                    u_θ = u_θ_data[θ_idx, φ_idx, r_idx]
                    u_φ = u_φ_data[θ_idx, φ_idx, r_idx]
                    
                    # Simple finite difference approximation for gradients
                    # In practice, this should use spectral derivatives
                    dC_dr = zero(T)  # Would compute radial gradient
                    dC_dθ = zero(T)  # Would compute θ gradient  
                    dC_dφ = zero(T)  # Would compute φ gradient
                    
                    # Advection: -u·∇C
                    adv_data[θ_idx, φ_idx, r_idx] = -(u_r * dC_dr + u_θ * dC_dθ + u_φ * dC_dφ)
                end
            end
        end
    end
end

function add_compositional_sources_local!(comp_field::SHTnsCompositionField{T}, 
                                          oc_domain::RadialDomain) where T
    # Add volumetric compositional sources (completely local operation)
    
    adv_data = parent(comp_field.advection_physical.data)
    
    if !all(iszero, comp_field.internal_sources)
        # Get local physical dimensions
        nlat, nlon, nr_local = size(adv_data)
        
        # Add sources (assumes sources are radially symmetric)
        for r_local in 1:nr_local
            for φ_idx in 1:nlon
                for θ_idx in 1:nlat
                    if r_local <= length(comp_field.internal_sources)
                        adv_data[θ_idx, φ_idx, r_local] += comp_field.internal_sources[r_local]
                    end
                end
            end
        end
    end
end

# Boundary conditions in spectral space
function apply_composition_boundary_conditions_spectral!(comp_field::SHTnsCompositionField{T}, 
                                                         oc_domain::RadialDomain) where T
    # Apply boundary conditions in spectral space
    
    spec_real = parent(comp_field.spectral.data_real)
    spec_imag = parent(comp_field.spectral.data_imag)
    
    lm_range = range_local(comp_field.config.pencils.spec, 1)
    r_range  = range_local(comp_field.config.pencils.spec, 3)
    
    # Check which boundaries are local
    has_inner = 1 in r_range
    has_outer = oc_domain.N in r_range
    
    if has_inner || has_outer
        for (local_lm, lm_idx) in enumerate(lm_range)
            bc_inner = comp_field.bc_type_inner[lm_idx]
            bc_outer = comp_field.bc_type_outer[lm_idx]
            
            # Apply inner boundary condition
            if has_inner
                r_local = 1
                if bc_inner == 1  # Fixed composition
                    spec_real[local_lm, 1, r_local] = comp_field.boundary_values[1, lm_idx]
                    spec_imag[local_lm, 1, r_local] = 0.0  # Fixed values are real
                elseif bc_inner == 2  # No-flux (zero gradient)
                    # For no-flux, we would set the gradient to zero
                    # Implementation depends on the specific scheme used
                end
            end
            
            # Apply outer boundary condition
            if has_outer
                r_local = oc_domain.N - first(r_range) + 1
                if r_local <= size(spec_real, 3)
                    if bc_outer == 1  # Fixed composition
                        spec_real[local_lm, 1, r_local] = comp_field.boundary_values[2, lm_idx]
                        spec_imag[local_lm, 1, r_local] = 0.0  # Fixed values are real
                    elseif bc_outer == 2  # No-flux (zero gradient)
                        # For no-flux, we would set the gradient to zero
                        # Implementation depends on the specific scheme used
                    end
                end
            end
        end
    end
end

# Diagnostic functions
function compute_composition_rms(comp_field::SHTnsCompositionField{T}, oc_domain::RadialDomain) where T
    # Compute RMS composition
    spec_real = parent(comp_field.spectral.data_real)
    spec_imag = parent(comp_field.spectral.data_imag)
    
    local_sum = zero(T)
    
    # Sum over local spectral modes
    for r_idx in axes(spec_real, 3)
        for lm_idx in axes(spec_real, 1)
            val_real = spec_real[lm_idx, 1, r_idx]
            val_imag = spec_imag[lm_idx, 1, r_idx]
            local_sum += val_real^2 + val_imag^2
        end
    end
    
    # Global reduction
    comm = get_comm()
    global_sum = MPI.Allreduce(local_sum, MPI.SUM, comm)
    
    return sqrt(global_sum / (oc_domain.N * comp_field.config.nlm))
end

function compute_composition_energy(comp_field::SHTnsCompositionField{T}, oc_domain::RadialDomain) where T
    # Compute compositional energy ∫ C² dV
    
    # Transform to physical space first
    shtnskit_spectral_to_physical!(comp_field.spectral, comp_field.work_physical)
    
    comp_data = parent(comp_field.work_physical.data)
    local_energy = zero(T)
    
    # Integrate over local physical space
    nlat, nlon, nr = size(comp_data)
    
    for r_idx in 1:nr
        for φ_idx in 1:nlon  
            for θ_idx in 1:nlat
                C = comp_data[θ_idx, φ_idx, r_idx]
                local_energy += C^2
            end
        end
    end
    
    # Global reduction
    comm = get_comm()
    global_energy = MPI.Allreduce(local_energy, MPI.SUM, comm)
    
    return global_energy / (comp_field.config.nlat * comp_field.config.nlon * oc_domain.N)
end

function get_composition_statistics(comp_field::SHTnsCompositionField{T}, oc_domain::RadialDomain) where T
    rms_comp = compute_composition_rms(comp_field, oc_domain)
    comp_energy = compute_composition_energy(comp_field, oc_domain)
    
    return (rms = rms_comp, energy = comp_energy)
end

# Initial condition functions  
function set_composition_ic!(comp_field::SHTnsCompositionField{T}, 
                             ic_type::Symbol, oc_domain::RadialDomain) where T
    # Set initial conditions for composition field
    
    spec_real = parent(comp_field.spectral.data_real)
    spec_imag = parent(comp_field.spectral.data_imag)
    
    if ic_type == :uniform
        # Uniform composition (only l=0, m=0 mode)
        fill!(spec_real, zero(T))
        fill!(spec_imag, zero(T))
        
        # Set uniform value in l=0, m=0 mode
        for r_idx in axes(spec_real, 3)
            spec_real[1, 1, r_idx] = 1.0  # Uniform composition = 1
        end
        
    elseif ic_type == :linear
        # Linear profile in radius
        for r_idx in axes(spec_real, 3)
            r = oc_domain.r[r_idx, 4]  # Normalized radius
            spec_real[1, 1, r_idx] = r  # Linear in radius
        end
        
    elseif ic_type == :random_perturbation
        # Small random perturbation on all modes
        fill!(spec_real, zero(T))
        fill!(spec_imag, zero(T))
        
        amplitude = 1e-3
        for r_idx in axes(spec_real, 3)
            for lm_idx in axes(spec_real, 1)
                if lm_idx > 1  # Don't perturb l=0, m=0 mode
                    spec_real[lm_idx, 1, r_idx] = amplitude * (rand(T) - 0.5)
                    spec_imag[lm_idx, 1, r_idx] = amplitude * (rand(T) - 0.5)
                end
            end
        end
    end
end

# Boundary condition setup
function set_composition_boundary_conditions!(comp_field::SHTnsCompositionField{T}, 
                                              bc_inner::Symbol, bc_outer::Symbol,
                                              value_inner::T = zero(T), value_outer::T = zero(T)) where T
    # Set boundary condition types and values
    
    if bc_inner == :fixed
        fill!(comp_field.bc_type_inner, 1)
        comp_field.boundary_values[1, :] .= value_inner
    elseif bc_inner == :no_flux
        fill!(comp_field.bc_type_inner, 2)
    end
    
    if bc_outer == :fixed  
        fill!(comp_field.bc_type_outer, 1)
        comp_field.boundary_values[2, :] .= value_outer
    elseif bc_outer == :no_flux
        fill!(comp_field.bc_type_outer, 2)
    end
end

# ============================================================================
# NetCDF Boundary Condition Integration for Composition
# ============================================================================

"""
    apply_netcdf_composition_boundaries!(comp_field::SHTnsCompositionField{T}, 
                                        boundary_set::BoundaryConditionSet{T},
                                        current_time::T=T(0)) where T

Apply compositional boundary conditions loaded from NetCDF files to the composition field.

# Arguments
- `comp_field`: SHTns composition field to apply boundaries to
- `boundary_set`: Boundary condition set loaded from NetCDF files
- `current_time`: Current simulation time (for time-dependent boundaries)

# Example
```julia
# Load boundaries from NetCDF files
comp_boundaries = load_composition_boundaries("cmb_composition.nc", "surface_composition.nc")

# Apply to composition field
apply_netcdf_composition_boundaries!(comp_field, comp_boundaries)
```
"""
function apply_netcdf_composition_boundaries!(comp_field::SHTnsCompositionField{T}, 
                                            boundary_set::BoundaryConditionSet{T},
                                            current_time::T=T(0)) where T
    
    if boundary_set.field_name != "composition"
        throw(ArgumentError("Boundary set field name '$(boundary_set.field_name)' != 'composition'"))
    end
    
    config = comp_field.config
    
    # Determine time indices for time-dependent boundaries
    inner_time_idx = get_boundary_time_index(boundary_set.inner_boundary, current_time)
    outer_time_idx = get_boundary_time_index(boundary_set.outer_boundary, current_time)
    
    # Create target coordinate grids from SHTns configuration
    target_theta, target_phi = create_shtns_coordinate_grids(config)
    
    # Interpolate boundary data to SHTns grid
    inner_values = interpolate_boundary_to_grid(boundary_set.inner_boundary, 
                                               target_theta, target_phi, inner_time_idx)
    outer_values = interpolate_boundary_to_grid(boundary_set.outer_boundary, 
                                               target_theta, target_phi, outer_time_idx)
    
    # Convert to spectral space and apply
    apply_composition_physical_boundaries_to_spectral!(comp_field, inner_values, outer_values, config)
    
    @info "Applied NetCDF composition boundaries (time=$(current_time))"
end

"""
    apply_composition_physical_boundaries_to_spectral!(comp_field, inner_values, outer_values, config)

Convert physical compositional boundary values to spectral coefficients and apply them.
"""
function apply_composition_physical_boundaries_to_spectral!(comp_field::SHTnsCompositionField{T},
                                                           inner_values::Matrix{T}, 
                                                           outer_values::Matrix{T},
                                                           config::SHTnsKitConfig) where T
    
    # Use work arrays for physical to spectral transforms
    phys_work = comp_field.work_physical
    spec_work = comp_field.work_spectral
    
    # Transform inner boundary to spectral space
    phys_data = parent(phys_work.data)
    nlat, nlon = size(inner_values)
    
    # Copy inner boundary data to work array (assumes matching grid)
    if size(phys_data, 1) >= nlat && size(phys_data, 2) >= nlon
        phys_data[1:nlat, 1:nlon, 1] .= inner_values
        
        # Transform to spectral space
        shtnskit_physical_to_spectral!(phys_work, spec_work)
        
        # Extract and apply inner boundary coefficients
        apply_composition_spectral_boundary_coefficients!(comp_field, spec_work, :inner)
    else
        @warn "Grid size mismatch for inner boundary: expected $(nlat)×$(nlon), got $(size(phys_data)[1:2])"
    end
    
    # Transform outer boundary to spectral space
    if size(phys_data, 1) >= nlat && size(phys_data, 2) >= nlon
        phys_data[1:nlat, 1:nlon, 1] .= outer_values
        
        # Transform to spectral space  
        shtnskit_physical_to_spectral!(phys_work, spec_work)
        
        # Extract and apply outer boundary coefficients
        apply_composition_spectral_boundary_coefficients!(comp_field, spec_work, :outer)
    else
        @warn "Grid size mismatch for outer boundary: expected $(nlat)×$(nlon), got $(size(phys_data)[1:2])"
    end
end

"""
    apply_composition_spectral_boundary_coefficients!(comp_field, spec_work, boundary::Symbol)

Apply spectral coefficients from boundary data to the composition field.
"""
function apply_composition_spectral_boundary_coefficients!(comp_field::SHTnsCompositionField{T},
                                                          spec_work::SHTnsSpectralField{T},
                                                          boundary::Symbol) where T
    
    spec_real_work = parent(spec_work.data_real)
    spec_imag_work = parent(spec_work.data_imag)
    
    lm_range = range_local(comp_field.config.pencils.spec, 1)
    boundary_idx = (boundary == :inner) ? 1 : 2
    
    # Copy spectral coefficients to boundary values array
    @inbounds for lm_idx in lm_range
        if lm_idx <= comp_field.config.nlm
            local_lm = lm_idx - first(lm_range) + 1
            
            if local_lm <= size(spec_real_work, 1) && size(spec_real_work, 3) >= 1
                comp_field.boundary_values[boundary_idx, lm_idx] = spec_real_work[local_lm, 1, 1]
                
                # Set boundary condition type to Dirichlet (fixed composition)
                if boundary == :inner
                    comp_field.bc_type_inner[lm_idx] = 1
                else
                    comp_field.bc_type_outer[lm_idx] = 1
                end
            end
        end
    end
end

"""
    update_composition_boundaries_from_netcdf!(comp_field::SHTnsCompositionField{T},
                                              boundary_set::BoundaryConditionSet{T},
                                              timestep::Int, dt::T) where T

Update compositional boundary conditions during time evolution.

# Arguments
- `comp_field`: Composition field to update
- `boundary_set`: NetCDF boundary condition set
- `timestep`: Current timestep number
- `dt`: Time step size

# Example
```julia
# During time stepping loop
for timestep in 1:nsteps
    current_time = timestep * dt
    update_composition_boundaries_from_netcdf!(comp_field, comp_boundaries, timestep, dt)
    # ... perform time step
end
```
"""
function update_composition_boundaries_from_netcdf!(comp_field::SHTnsCompositionField{T},
                                                   boundary_set::BoundaryConditionSet{T},
                                                   timestep::Int, dt::T) where T
    
    current_time = timestep * dt
    apply_netcdf_composition_boundaries!(comp_field, boundary_set, current_time)
end

"""
    validate_netcdf_composition_compatibility(boundary_set::BoundaryConditionSet, config::SHTnsKitConfig)

Validate that NetCDF composition boundaries are compatible with the SHTns configuration.
"""
function validate_netcdf_composition_compatibility(boundary_set::BoundaryConditionSet, 
                                                  config::SHTnsKitConfig)
    errors = String[]
    
    # Check grid compatibility
    if boundary_set.inner_boundary.nlat != config.nlat
        push!(errors, "Inner boundary nlat ($(boundary_set.inner_boundary.nlat)) != config nlat ($(config.nlat))")
    end
    
    if boundary_set.inner_boundary.nlon != config.nlon
        push!(errors, "Inner boundary nlon ($(boundary_set.inner_boundary.nlon)) != config nlon ($(config.nlon))")
    end
    
    if boundary_set.outer_boundary.nlat != config.nlat
        push!(errors, "Outer boundary nlat ($(boundary_set.outer_boundary.nlat)) != config nlat ($(config.nlat))")
    end
    
    if boundary_set.outer_boundary.nlon != config.nlon
        push!(errors, "Outer boundary nlon ($(boundary_set.outer_boundary.nlon)) != config nlon ($(config.nlon))")
    end
    
    if !isempty(errors)
        error_msg = "NetCDF composition boundary validation failed:\n" * join(errors, "\n")
        throw(ArgumentError(error_msg))
    end
    
    @info "NetCDF composition boundaries are compatible with SHTns configuration"
end

# Note: Helper functions like get_boundary_time_index and create_shtns_coordinate_grids
# are defined in thermal.jl and will be available through the main module exports
