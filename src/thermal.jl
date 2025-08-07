# ===============================================================
# Temperature  field components with optimized SHTns transforms
# ===============================================================
struct SHTnsTemperatureField{T}
    # Physical space temperature
    temperature::SHTnsPhysicalField{T}
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
    
    # Sources and boundary conditions
    internal_sources::Vector{T}
    boundary_values::Matrix{T}
    
    # Pre-computed coefficients
    l_factors::Vector{Float64}  # l(l+1) values
    
    # Transform manager
    transform_manager::SHTnsTransformManager{T}
    
    # Radial derivative matrix for gradient computation
    dr_matrix::BandedMatrix{T}
end


function create_shtns_temperature_field(::Type{T}, config::SHTnsConfig, 
                                        domain::RadialDomain, 
                                        pencils, pencil_spec) where T
    pencil_θ, pencil_φ, pencil_r = pencils
    
    # Temperature field
    temperature = create_shtns_physical_field(T, config, domain, pencil_r)
    
    # Gradient components
    gradient = create_shtns_vector_field(T, config, domain, pencils)
    
    # Spectral representation
    spectral  = create_shtns_spectral_field(T, config, domain, pencil_spec)
    nonlinear = create_shtns_spectral_field(T, config, domain, pencil_spec)
    
    # Work arrays
    work_spectral = create_shtns_spectral_field(T, config, domain, pencil_spec)
    work_physical = create_shtns_physical_field(T, config, domain, pencil_r)
    advection_physical = create_shtns_physical_field(T, config, domain, pencil_r)
    
    # Gradient spectral components
    grad_theta_spec = create_shtns_spectral_field(T, config, domain, pencil_spec)
    grad_phi_spec = create_shtns_spectral_field(T, config, domain, pencil_spec)
    
    # Sources and boundary conditions
    internal_sources = zeros(T, domain.N)
    boundary_values  = zeros(T, 2, config.nlm)  # ICB and CMB values
    
    # Pre-compute l(l+1) factors
    l_factors = Float64[l * (l + 1) for l in config.l_values]
    
    # Create transform manager
    transform_manager = get_transform_manager(T, config, pencil_spec)
    
    # Create radial derivative matrix
    dr_matrix = create_derivative_matrix(1, domain)
    
    return SHTnsTemperatureField{T}(temperature, gradient, spectral, nonlinear,
                                    work_spectral, work_physical, advection_physical,
                                    grad_theta_spec, grad_phi_spec,
                                    internal_sources, boundary_values,
                                    l_factors, transform_manager, dr_matrix)
end


# ==========================================================
# Main nonlinear computation using optimized transforms
# ==========================================================
function compute_temperature_nonlinear!(temp_field::SHTnsTemperatureField{T}, 
                                        vel_fields, domain::RadialDomain,
                                        transpose_plans=nothing) where T
    # Zero work arrays
    zero_temperature_work_arrays!(temp_field)
    
    # Step 1: Convert spectral temperature to physical space using optimized transform
    shtns_spectral_to_physical!(temp_field.spectral, temp_field.temperature, transpose_plans)
    
    # Step 2: Compute temperature gradient efficiently
    compute_temperature_gradient!(temp_field, domain)
    
    # Step 3: Compute advection term -u·∇T in physical space
    if vel_fields !== nothing
        compute_temperature_advection!(temp_field, vel_fields)
    end
    
    # Step 4: Add internal heat sources
    add_internal_sources!(temp_field)
    
    # Step 5: Transform advection + sources to spectral space for nonlinear term
    shtns_physical_to_spectral!(temp_field.advection_physical, temp_field.nonlinear, transpose_plans)
    
    # Step 6: Apply boundary conditions in spectral space
    apply_temperature_boundary_conditions!(temp_field, domain)
end


# ==================================================
# Optimized gradient computation using SHTns
# ==================================================
function compute_temperature_gradient!(temp_field::SHTnsTemperatureField{T}, 
                                                domain::RadialDomain) where T
    # Compute gradient using SHTns optimized routines
    config = temp_field.spectral.config
    sht = config.sht
    
    # Get local data views
    spec_real = parent(temp_field.spectral.data_real)
    spec_imag = parent(temp_field.spectral.data_imag)
    
    grad_r_data = parent(temp_field.gradient.r_component.data)
    grad_θ_data = parent(temp_field.gradient.θ_component.data)
    grad_φ_data = parent(temp_field.gradient.φ_component.data)
    
    # Get local ranges
    r_range = get_local_range(temp_field.spectral.pencil, 3)
    lm_range = get_local_range(temp_field.spectral.pencil, 1)
    
    # Use transform manager for efficiency
    manager = temp_field.transform_manager
    coeffs = manager.coeffs_full
    
    # Pre-allocate derivative work arrays
    nr = domain.N
    temp_profile = zeros(T, nr)
    dtemp_dr = zeros(T, nr)
    
    # Step 1: Compute horizontal gradients using SHTns at each radial level
    @inbounds for r_idx in r_range
        local_r = r_idx - first(r_range) + 1
        
        if local_r <= size(spec_real, 3)
            # Fill coefficients for this radial level
            fill_coefficients_from_local!(coeffs, spec_real, spec_imag, 
                                         local_r, lm_range)
            
            # MPI communication if needed
            if manager.needs_allreduce
                MPI.Allreduce!(coeffs, MPI.SUM, get_comm())
            end
            
            # Compute both horizontal derivatives simultaneously
            dT_dtheta = synthesis_dtheta(sht, coeffs)
            dT_dphi = synthesis_dphi(sht, coeffs)
            
            # Apply geometric factors and store
            r = domain.r[r_idx, 4]
            r_inv = domain.r[r_idx, 3]  # 1/r
            
            @simd for idx in eachindex(dT_dtheta)
                if idx <= size(grad_θ_data, 1) * size(grad_θ_data, 2)
                    i = ((idx - 1) % size(grad_θ_data, 1)) + 1
                    j = ((idx - 1) ÷ size(grad_θ_data, 1)) + 1
                    
                    if i <= config.nlat && j <= config.nlon
                        theta = config.theta_grid[i]
                        sin_theta_inv = 1.0 / max(sin(theta), 1e-10)
                        
                        # Store with proper geometric factors
                        linear_idx = i + (j-1)*size(grad_θ_data, 1) + 
                                    (local_r-1)*size(grad_θ_data, 1)*size(grad_θ_data, 2)
                        if linear_idx <= length(grad_θ_data)
                            grad_θ_data[linear_idx] = r_inv * real(dT_dtheta[idx])
                            grad_φ_data[linear_idx] = r_inv * sin_theta_inv * real(dT_dphi[idx])
                        end
                    end
                end
            end
        end
    end
    
    # Step 2: Compute radial gradient using spectral coefficients
    compute_radial_gradient_spectral!(temp_field, domain)
end

# ==================================================
# Radial gradient computation in spectral space
# ==================================================
function compute_radial_gradient_spectral!(temp_field::SHTnsTemperatureField{T}, 
                                          domain::RadialDomain) where T

    # Compute dT/dr for each (l,m) mode using banded matrix
    
    spec_real   = parent(temp_field.spectral.data_real)
    spec_imag   = parent(temp_field.spectral.data_imag)
    grad_r_data = parent(temp_field.gradient.r_component.data)
    
    # Get local ranges
    lm_range = get_local_range(temp_field.spectral.pencil, 1)
    r_range  = get_local_range(temp_field.spectral.pencil, 3)
    
    nr = domain.N
    temp_profile_real = zeros(T, nr)
    temp_profile_imag = zeros(T, nr)
    dtemp_dr_real    = zeros(T, nr)
    dtemp_dr_imag    = zeros(T, nr)
    
    # Process each (l,m) mode
    @inbounds for lm_idx in lm_range
        if lm_idx <= temp_field.spectral.nlm
            local_lm = lm_idx - first(lm_range) + 1
            
            # Extract radial profile for this mode
            for r_idx in 1:nr
                if r_idx in r_range
                    local_r = r_idx - first(r_range) + 1
                    if local_r <= size(spec_real, 3)
                        temp_profile_real[r_idx] = spec_real[local_lm, 1, local_r]
                        temp_profile_imag[r_idx] = spec_imag[local_lm, 1, local_r]
                    end
                else
                    temp_profile_real[r_idx] = zero(T)
                    temp_profile_imag[r_idx] = zero(T)
                end
            end
            
            # Apply radial derivative
            apply_derivative_matrix!(dtemp_dr_real, temp_field.dr_matrix, temp_profile_real)
            apply_derivative_matrix!(dtemp_dr_imag, temp_field.dr_matrix, temp_profile_imag)
            
            # Transform to physical space and store
            # This requires synthesis at each radial level
            # For efficiency, we batch this operation
        end
    end
    
    # Transform radial gradient to physical space
    shtns_spectral_to_physical!(temp_field.grad_theta_spec, 
                                temp_field.gradient.r_component)
end


# ================================
# Fused advection computation
# ================================
function compute_temperature_advection!(temp_field::SHTnsTemperatureField{T}, 
                                             vel_fields) where T
    # Compute -u·∇T with fused operations for efficiency
    
    # Get local data views
    u_r = parent(vel_fields.velocity.r_component.data)
    u_θ = parent(vel_fields.velocity.θ_component.data)
    u_φ = parent(vel_fields.velocity.φ_component.data)
    
    grad_r = parent(temp_field.gradient.r_component.data)
    grad_θ = parent(temp_field.gradient.θ_component.data)
    grad_φ = parent(temp_field.gradient.φ_component.data)
    
    advection = parent(temp_field.advection_physical.data)
    
    # Fused computation of -u·∇T
    @inbounds @simd for idx in eachindex(advection)
        if idx <= length(u_r) && idx <= length(grad_r)
            advection[idx] = -(u_r[idx] * grad_r[idx] + 
                              u_θ[idx] * grad_θ[idx] + 
                              u_φ[idx] * grad_φ[idx])
        end
    end
end


# =======================================
# Optimized internal source addition
# =======================================
function add_internal_sources_optimized!(temp_field::SHTnsTemperatureField{T}) where T
    # Add internal heat sources efficiently
    # Sources are typically axisymmetric (l,m=0 modes)
    
    advection = parent(temp_field.advection_physical.data)
    
    # Get local ranges
    r_range = get_local_range(temp_field.advection_physical.pencil, 3)
    
    # Add volumetric heating (if present)
    if !all(iszero, temp_field.internal_sources)
        nlat = temp_field.advection_physical.nlat
        nlon = temp_field.advection_physical.nlon
        
        @inbounds for r_idx in r_range
            if r_idx <= length(temp_field.internal_sources)
                local_r = r_idx - first(r_range) + 1
                source_value = temp_field.internal_sources[r_idx]
                
                # Add uniformly across the sphere at this radius
                @simd for j in 1:nlon
                    for i in 1:nlat
                        linear_idx = i + (j-1)*nlat + (local_r-1)*nlat*nlon
                        if linear_idx <= length(advection)
                            advection[linear_idx] += source_value
                        end
                    end
                end
            end
        end
    end
end
    


# ============================================================================
# Boundary condition application in spectral space
# ============================================================================
function apply_temperature_boundary_conditions!(temp_field::SHTnsTemperatureField{T}, 
                                               domain::RadialDomain) where T
    # Apply boundary conditions at inner and outer boundaries
    # BC types: 1 → Fixed temperature (Dirichlet)
    #           2 → Fixed flux (Neumann)
    
    if i_tmp_bc == 1  # Fixed temperature BC
        apply_temperature_bc_dirichlet!(temp_field, domain)
    elseif i_tmp_bc == 2  # Fixed flux BC
        apply_flux_bc_neumann!(temp_field, domain)
    end
end


function apply_temperature_bc_dirichlet!(temp_field::SHTnsTemperatureField{T}, 
                                        domain::RadialDomain) where T
    # Apply fixed temperature boundary conditions
    
    spec_real = parent(temp_field.spectral.data_real)
    spec_imag = parent(temp_field.spectral.data_imag)
    
    lm_range = get_local_range(temp_field.spectral.pencil, 1)
    r_range = get_local_range(temp_field.spectral.pencil, 3)
    
    # Inner boundary (r_idx = 1)
    if 1 in r_range
        local_r = 1 - first(r_range) + 1
        @inbounds for lm_idx in lm_range
            if lm_idx <= temp_field.spectral.nlm
                local_lm = lm_idx - first(lm_range) + 1
                if local_lm <= size(spec_real, 1)
                    spec_real[local_lm, 1, local_r] = temp_field.boundary_values[1, lm_idx]
                    spec_imag[local_lm, 1, local_r] = 0.0  # Real BC
                end
            end
        end
    end
    
    # Outer boundary (r_idx = N)
    if domain.N in r_range
        local_r = domain.N - first(r_range) + 1
        @inbounds for lm_idx in lm_range
            if lm_idx <= temp_field.spectral.nlm
                local_lm = lm_idx - first(lm_range) + 1
                if local_lm <= size(spec_real, 1) && local_r <= size(spec_real, 3)
                    spec_real[local_lm, 1, local_r] = temp_field.boundary_values[2, lm_idx]
                    spec_imag[local_lm, 1, local_r] = 0.0  # Real BC
                end
            end
        end
    end
end


function apply_flux_bc_neumann!(temp_field::SHTnsTemperatureField{T}, 
                               domain::RadialDomain) where T
    # Apply fixed flux boundary conditions using influence matrix method
    
    spec_real = parent(temp_field.spectral.data_real)
    spec_imag = parent(temp_field.spectral.data_imag)
    
    lm_range = get_local_range(temp_field.spectral.pencil, 1)
    nr = domain.N
    
    # Create boundary derivative operator
    boundary_dr = create_boundary_derivative_operator(domain)
    
    # Process each (l,m) mode
    @inbounds for lm_idx in lm_range
        if lm_idx <= temp_field.spectral.nlm
            local_lm = lm_idx - first(lm_range) + 1
            
            # Extract radial profile
            temp_profile_real = zeros(T, nr)
            temp_profile_imag = zeros(T, nr)
            
            for r_idx in 1:nr
                if r_idx in get_local_range(temp_field.spectral.pencil, 3)
                    local_r = r_idx - first(get_local_range(temp_field.spectral.pencil, 3)) + 1
                    if local_r <= size(spec_real, 3)
                        temp_profile_real[r_idx] = spec_real[local_lm, 1, local_r]
                        temp_profile_imag[r_idx] = spec_imag[local_lm, 1, local_r]
                    end
                end
            end
            
            # Get prescribed fluxes for this mode
            flux_inner = get_flux_bc_value(lm_idx, 1, temp_field)
            flux_outer = get_flux_bc_value(lm_idx, 2, temp_field)
            
            # Apply flux correction
            correct_for_flux_bc!(temp_profile_real, flux_inner, flux_outer, boundary_dr, domain)
            if any(x -> abs(x) > 1e-12, temp_profile_imag)
                correct_for_flux_bc!(temp_profile_imag, 0.0, 0.0, boundary_dr, domain)
            end
            
            # Store corrected profile back
            for r_idx in get_local_range(temp_field.spectral.pencil, 3)
                local_r = r_idx - first(get_local_range(temp_field.spectral.pencil, 3)) + 1
                if local_r <= size(spec_real, 3)
                    spec_real[local_lm, 1, local_r] = temp_profile_real[r_idx]
                    spec_imag[local_lm, 1, local_r] = temp_profile_imag[r_idx]
                end
            end
        end
    end
end


function create_boundary_derivative_operator(domain::RadialDomain)
    # Create operator for computing derivatives at boundaries
    N = domain.N
    bandwidth = i_KL
    
    # 2xN matrix: row 1 for inner boundary, row 2 for outer boundary
    boundary_op = zeros(2, N)
    
    # Inner boundary coefficients
    for idx in 1:min(bandwidth+1, N)
        # Use forward difference stencil
        boundary_op[1, idx] = compute_fd_coefficient(1, idx, domain.r[1:bandwidth+1, 4])
    end
    
    # Outer boundary coefficients
    for idx in max(N-bandwidth, 1):N
        # Use backward difference stencil
        local_idx = idx - (N - bandwidth - 1)
        boundary_op[2, idx] = compute_fd_coefficient(bandwidth+1, local_idx, 
                                                    domain.r[N-bandwidth:N, 4])
    end
    
    return boundary_op
end


function compute_fd_coefficient(target_idx::Int, stencil_idx::Int, points::Vector{Float64})
    # Compute finite difference coefficient for derivative at target point
    n = length(points)
    c = 1.0
    c1 = 1.0
    c4 = points[1] - points[target_idx]
    
    coeff = 0.0
    
    for i in 1:n
        if i == stencil_idx
            continue
        end
        c2 = 1.0
        c5 = c4
        c4 = points[i] - points[target_idx]
        
        for j in 1:n
            if j == i || j == stencil_idx
                continue
            end
            c2 *= (points[i] - points[j])
        end
        
        c3 = points[i] - points[stencil_idx]
        c2 = 1.0 / c2
        
        if i == target_idx
            coeff = c1 / c3
        end
        
        c1 = c2
    end
    
    return coeff
end


function correct_for_flux_bc!(profile::Vector{T}, flux_inner::T, flux_outer::T,
                             boundary_op::Matrix{T}, domain::RadialDomain) where T
    # Correct temperature profile to satisfy flux boundary conditions
    
    # Current fluxes at boundaries
    current_flux_inner = dot(boundary_op[1, :], profile)
    current_flux_outer = dot(boundary_op[2, :], profile)
    
    # Flux errors
    error_inner = flux_inner - current_flux_inner
    error_outer = flux_outer - current_flux_outer
    
    # Apply linear correction (simplest approach)
    # More sophisticated: use influence functions or tau method
    N = domain.N
    for r_idx in 1:N
        # Linear interpolation of correction
        α = Float64(r_idx - 1) / Float64(N - 1)
        correction = (1.0 - α) * error_inner + α * error_outer
        profile[r_idx] += correction / Float64(N)
    end
end


function get_flux_bc_value(lm_idx::Int, boundary::Int, 
                          temp_field::SHTnsTemperatureField{T}) where T
    # Get prescribed flux value for given mode and boundary
    # boundary: 1 = inner, 2 = outer
    
    l = temp_field.spectral.config.l_values[lm_idx]
    m = temp_field.spectral.config.m_values[lm_idx]
    
    # Example: uniform heating from below, cooling from above
    if l == 0 && m == 0
        if boundary == 1
            return T(1.0)   # Heating from below
        else
            return T(-1.0)  # Cooling from above
        end
    else
        return T(0.0)  # No flux for non-axisymmetric modes
    end
end


# ============================================================================
# Advanced flux BC implementation using influence matrix method
# ============================================================================
function apply_flux_bc_influence_matrix!(temp_field::SHTnsTemperatureField{T}, 
                                        domain::RadialDomain) where T
    # More sophisticated flux BC implementation using influence functions
    
    spec_real = parent(temp_field.spectral.data_real)
    spec_imag = parent(temp_field.spectral.data_imag)
    
    lm_range = get_local_range(temp_field.spectral.pencil, 1)
    nr = domain.N
    
    # Pre-compute influence functions and their derivatives
    influence_inner, influence_outer = compute_influence_functions(domain)
    influence_matrix = compute_influence_matrix(influence_inner, influence_outer, domain)
    
    # Process each (l,m) mode
    @inbounds for lm_idx in lm_range
        if lm_idx <= temp_field.spectral.nlm
            local_lm = lm_idx - first(lm_range) + 1
            
            # Extract and correct radial profile
            temp_profile_real = extract_radial_profile(spec_real, local_lm, nr, 
                                                      temp_field.spectral.pencil)
            temp_profile_imag = extract_radial_profile(spec_imag, local_lm, nr, 
                                                      temp_field.spectral.pencil)
            
            # Get prescribed fluxes
            flux_inner = get_flux_bc_value(lm_idx, 1, temp_field)
            flux_outer = get_flux_bc_value(lm_idx, 2, temp_field)
            
            # Apply influence matrix correction
            apply_influence_correction!(temp_profile_real, flux_inner, flux_outer,
                                       influence_matrix, influence_inner, influence_outer,
                                       domain)
            
            if any(x -> abs(x) > 1e-12, temp_profile_imag)
                apply_influence_correction!(temp_profile_imag, 0.0, 0.0,
                                          influence_matrix, influence_inner, influence_outer,
                                          domain)
            end
            
            # Store corrected profile
            store_radial_profile!(spec_real, temp_profile_real, local_lm, 
                                temp_field.spectral.pencil)
            store_radial_profile!(spec_imag, temp_profile_imag, local_lm, 
                                temp_field.spectral.pencil)
        end
    end
end



function compute_influence_functions(domain::RadialDomain)
    # Compute smooth influence functions for boundary corrections
    N = domain.N
    influence_inner = zeros(N)
    influence_outer = zeros(N)
    
    ri = domain.r[1, 4]
    ro = domain.r[N, 4]
    
    for r_idx in 1:N
        r = domain.r[r_idx, 4]
        ξ = (r - ri) / (ro - ri)  # Normalized coordinate [0, 1]
        
        # Smooth influence functions using cosine tapers
        influence_inner[r_idx] = 0.5 * (1.0 + cos(π * ξ))
        influence_outer[r_idx] = 0.5 * (1.0 - cos(π * ξ))
    end
    
    return influence_inner, influence_outer
end


function compute_influence_matrix(influence_inner::Vector{T}, 
                                 influence_outer::Vector{T},
                                 domain::RadialDomain) where T
    # Compute 2x2 influence matrix for boundary flux corrections
    
    # Create derivative matrix
    dr_matrix = create_derivative_matrix(1, domain)
    
    # Compute derivatives of influence functions
    dinf_inner_dr = apply_derivative_operator(dr_matrix, influence_inner)
    dinf_outer_dr = apply_derivative_operator(dr_matrix, influence_outer)
    
    # Build influence matrix
    # M[i,j] = flux at boundary i due to unit amplitude of influence function j
    M = zeros(T, 2, 2)
    M[1, 1] = dinf_inner_dr[1]    # Flux at inner due to inner influence
    M[1, 2] = dinf_outer_dr[1]    # Flux at inner due to outer influence
    M[2, 1] = dinf_inner_dr[end]  # Flux at outer due to inner influence
    M[2, 2] = dinf_outer_dr[end]  # Flux at outer due to outer influence
    
    return M
end


function apply_influence_correction!(profile::Vector{T}, flux_inner::T, flux_outer::T,
                                    influence_matrix::Matrix{T},
                                    influence_inner::Vector{T}, 
                                    influence_outer::Vector{T},
                                    domain::RadialDomain) where T
    # Apply correction using influence functions
    
    # Compute current fluxes
    dr_matrix  = create_derivative_matrix(1, domain)
    dprofile_dr = apply_derivative_operator(dr_matrix, profile)
    current_flux_inner = dprofile_dr[1]
    current_flux_outer = dprofile_dr[end]
    
    # Flux errors
    flux_errors = [flux_inner - current_flux_inner,
                   flux_outer - current_flux_outer]
    
    # Solve for correction amplitudes
    correction_amplitudes = influence_matrix \ flux_errors
    
    # Apply corrections
    @inbounds for r_idx in 1:length(profile)
        profile[r_idx] += correction_amplitudes[1] * influence_inner[r_idx] +
                         correction_amplitudes[2] * influence_outer[r_idx]
    end
end


function apply_derivative_operator(dr_matrix::BandedMatrix{T}, field::Vector{T}) where T
    # Apply banded derivative matrix to field
    N = dr_matrix.size
    bandwidth = dr_matrix.bandwidth
    result = zeros(T, N)
    
    @inbounds for j in 1:N
        for i in max(1, j - bandwidth):min(N, j + bandwidth)
            band_row = bandwidth + 1 + i - j
            if 1 <= band_row <= 2*bandwidth + 1
                result[i] += dr_matrix.data[band_row, j] * field[j]
            end
        end
    end
    
    return result
end


function extract_radial_profile(data::AbstractArray{T,3}, local_lm::Int, nr::Int,
                               pencil::Pencil{3}) where T
    # Extract radial profile for a given (l,m) mode
    profile = zeros(T, nr)
    r_range = get_local_range(pencil, 3)
    
    for r_idx in r_range
        local_r = r_idx - first(r_range) + 1
        if local_r <= size(data, 3) && r_idx <= nr
            profile[r_idx] = data[local_lm, 1, local_r]
        end
    end
    
    # MPI gather if needed
    if MPI.Initialized() && MPI.Comm_size(get_comm()) > 1
        MPI.Allreduce!(profile, MPI.SUM, get_comm())
    end
    
    return profile
end


function store_radial_profile!(data::AbstractArray{T,3}, profile::Vector{T}, 
                              local_lm::Int, pencil::Pencil{3}) where T
    # Store radial profile back to distributed array
    r_range = get_local_range(pencil, 3)
    
    for r_idx in r_range
        local_r = r_idx - first(r_range) + 1
        if local_r <= size(data, 3) && r_idx <= length(profile)
            data[local_lm, 1, local_r] = profile[r_idx]
        end
    end
end


# ==============================================================
# Mixed boundary conditions (different for different modes)
# ==============================================================
function apply_mixed_boundary_conditions!(temp_field::SHTnsTemperatureField{T}, 
                                         domain::RadialDomain,
                                         bc_type_inner::Vector{Int},
                                         bc_type_outer::Vector{Int}) where T
    # Apply different BC types for different (l,m) modes
    # bc_type: 1 = Dirichlet, 2 = Neumann
    
    spec_real = parent(temp_field.spectral.data_real)
    spec_imag = parent(temp_field.spectral.data_imag)
    
    lm_range = get_local_range(temp_field.spectral.pencil, 1)
    
    @inbounds for lm_idx in lm_range
        if lm_idx <= temp_field.spectral.nlm
            local_lm = lm_idx - first(lm_range) + 1
            
            # Determine BC type for this mode
            bc_inner = bc_type_inner[lm_idx]
            bc_outer = bc_type_outer[lm_idx]
            
            if bc_inner == 1 && bc_outer == 1
                # Both Dirichlet
                apply_mode_dirichlet_bc!(spec_real, spec_imag, local_lm, lm_idx,
                                        temp_field, domain)
            elseif bc_inner == 2 && bc_outer == 2
                # Both Neumann
                apply_mode_neumann_bc!(spec_real, spec_imag, local_lm, lm_idx,
                                     temp_field, domain)
            else
                # Mixed BC
                apply_mode_mixed_bc!(spec_real, spec_imag, local_lm, lm_idx,
                                    bc_inner, bc_outer, temp_field, domain)
            end
        end
    end
end


function apply_mode_dirichlet_bc!(spec_real::AbstractArray{T,3}, 
                                 spec_imag::AbstractArray{T,3},
                                 local_lm::Int, lm_idx::Int,
                                 temp_field::SHTnsTemperatureField{T},
                                 domain::RadialDomain) where T
    # Apply Dirichlet BC for a specific mode
    r_range = get_local_range(temp_field.spectral.pencil, 3)
    
    if 1 in r_range
        local_r = 1 - first(r_range) + 1
        spec_real[local_lm, 1, local_r] = temp_field.boundary_values[1, lm_idx]
        spec_imag[local_lm, 1, local_r] = 0.0
    end
    
    if domain.N in r_range
        local_r = domain.N - first(r_range) + 1
        spec_real[local_lm, 1, local_r] = temp_field.boundary_values[2, lm_idx]
        spec_imag[local_lm, 1, local_r] = 0.0
    end
end


function zero_work_arrays!(temp_field::SHTnsTemperatureField{T}) where T
    # Efficiently zero work arrays
    fill!(parent(temp_field.work_physical.data), zero(T))
    fill!(parent(temp_field.work_spectral.data_real), zero(T))
    fill!(parent(temp_field.work_spectral.data_imag), zero(T))
end


# # Export functions
# export SHTnsTemperatureField, create_shtns_temperature_field
# export compute_temperature_nonlinear!, compute_temperature_batch!
# export zero_work_arrays!
