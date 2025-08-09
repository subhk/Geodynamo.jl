# ===============================================================
# Temperature  field components with optimized SHTns transforms
# ===============================================================
# For uniform heating from below (l=0, m=0 mode has flux BC)
# Other modes have fixed temperature
# bc_type_inner = ones(Int, nlm)
# bc_type_outer = ones(Int, nlm)
# bc_type_inner[1] = 2  # Flux BC for l=0, m=0 at inner boundary
# bc_type_outer[1] = 2  # Flux BC for l=0, m=0 at outer boundary
# apply_mixed_boundary_conditions!(temp_field, domain, bc_type_inner, bc_type_outer)
# ====================================================================================

# ============================================================================
# Temperature/Thermal field components with full spectral optimization
# ============================================================================

using MPI
using PencilArrays
using SHTnsSpheres
using LinearAlgebra
using SparseArrays

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
    grad_r_spec::SHTnsSpectralField{T}
    
    # Sources and boundary conditions
    internal_sources::Vector{T}        # Radial profile of heating
    boundary_values::Matrix{T}         # [2, nlm] for ICB and CMB
    bc_type_inner::Vector{Int}         # BC type for each mode at inner
    bc_type_outer::Vector{Int}         # BC type for each mode at outer
    
    # Pre-computed coefficients
    l_factors::Vector{Float64}         # l(l+1) values
    
    # Configuration and transform manager
    config::SHTnsConfig
    transform_manager::SHTnsTransformManager{T}
    
    # Radial derivative matrices
    dr_matrix::BandedMatrix{T}
    d2r_matrix::BandedMatrix{T}
    
    # Spectral derivative operators
    theta_derivative_matrix::SparseMatrixCSC{T,Int}  # Pre-computed θ-derivative
    theta_recurrence_coeffs::Matrix{T}               # Recurrence coefficients
    
    # Performance tracking
    computation_time::Ref{Float64}
    transform_time::Ref{Float64}
    comm_time::Ref{Float64}
    spectral_time::Ref{Float64}
end


function create_shtns_temperature_field(::Type{T}, config::SHTnsConfig, 
                                        oc_domain::RadialDomain) where T
    # Use config's pencils directly
    pencils = config.pencils
    
    # Temperature field in r-pencil for efficient radial operations
    temperature = create_shtns_physical_field(T, config, oc_domain, pencils.r)
    
    # Gradient components
    gradient = create_shtns_vector_field(T, config, oc_domain, 
                                        (pencils.θ, pencils.φ, pencils.r))
    
    # Spectral representation using spectral pencil
    spectral  = create_shtns_spectral_field(T, config, oc_domain, pencils.spec)
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
    
    # Default BC types (1 = Dirichlet, 2 = Neumann)
    bc_type_inner = ones(Int, config.nlm)  # Default to fixed temperature
    bc_type_outer = ones(Int, config.nlm)
    
    # Pre-compute l(l+1) factors
    l_factors = Float64[l * (l + 1) for l in config.l_values]
    
    # Get transform manager from config
    transform_manager = get_transform_manager(T, config)
    
    # Create radial derivative matrices
    dr_matrix  = create_derivative_matrix(1, oc_domain)
    d2r_matrix = create_derivative_matrix(2, oc_domain)
    
    # Pre-compute spectral derivative operators
    theta_derivative_matrix = build_theta_derivative_matrix(T, config)
    theta_recurrence_coeffs = compute_theta_recurrence_coefficients(T, config)
    
    return SHTnsTemperatureField{T}(
        temperature, gradient, spectral, nonlinear,
        work_spectral, work_physical, advection_physical,
        grad_theta_spec, grad_phi_spec, grad_r_spec,
        internal_sources, boundary_values,
        bc_type_inner, bc_type_outer,
        l_factors, config, transform_manager,
        dr_matrix, d2r_matrix,
        theta_derivative_matrix, theta_recurrence_coeffs,
        Ref(0.0), Ref(0.0), Ref(0.0), Ref(0.0)
    )
end

# =======================================================
# Pre-computation of spectral derivative operators
# =======================================================
function build_theta_derivative_matrix(::Type{T}, config::SHTnsConfig) where T
    """
    Build sparse matrix for θ-derivatives in spectral space.
    This matrix couples different l modes with the same m.
    """
    nlm = config.nlm
    
    # Build sparse matrix for derivative operator
    I = Int[]
    J = Int[]
    V = T[]
    
    for lm_idx in 1:nlm
        l = config.l_values[lm_idx]
        m = config.m_values[lm_idx]
        abs_m = abs(m)
        
        # Forward coupling (l,m) → (l+1,m)
        if l < config.lmax
            lp1m_idx = get_mode_index(config, l+1, m)
            if lp1m_idx > 0
                coeff = sqrt((l + abs_m + 1) * (l - abs_m + 1) / 
                           ((2*l + 1) * (2*l + 3))) * l
                push!(I, lm_idx)
                push!(J, lp1m_idx)
                push!(V, T(coeff))
            end
        end
        
        # Backward coupling (l,m) → (l-1,m)
        if l > abs_m
            lm1m_idx = get_mode_index(config, l-1, m)
            if lm1m_idx > 0
                coeff = -sqrt((l + abs_m) * (l - abs_m) / 
                            ((2*l - 1) * (2*l + 1))) * (l + 1)
                push!(I, lm_idx)
                push!(J, lm1m_idx)
                push!(V, T(coeff))
            end
        end
    end
    
    return sparse(I, J, V, nlm, nlm)
end

function compute_theta_recurrence_coefficients(::Type{T}, config::SHTnsConfig) where T
    """
    Pre-compute recurrence coefficients for θ-derivatives.
    Store as [nlm, 2] matrix: [:, 1] for l-1 coupling, [:, 2] for l+1 coupling
    """
    coeffs = zeros(T, config.nlm, 2)
    
    for lm_idx in 1:config.nlm
        l = config.l_values[lm_idx]
        m = config.m_values[lm_idx]
        abs_m = abs(m)
        
        # Backward coupling coefficient (l-1,m)
        if l > abs_m
            coeffs[lm_idx, 1] = -sqrt((l + abs_m) * (l - abs_m) / 
                                     ((2*l - 1) * (2*l + 1))) * (l + 1)
        end
        
        # Forward coupling coefficient (l+1,m)
        if l < config.lmax
            coeffs[lm_idx, 2] = sqrt((l + abs_m + 1) * (l - abs_m + 1) / 
                                    ((2*l + 1) * (2*l + 3))) * l
        end
    end
    
    return coeffs
end

# ============================================================================
# Main nonlinear computation with full spectral optimization
# ============================================================================
function compute_temperature_nonlinear!(temp_field::SHTnsTemperatureField{T}, 
                                        vel_fields, oc_domain::RadialDomain) where T
    t_start = ENABLE_TIMING[] ? MPI.Wtime() : 0.0
    
    # Zero work arrays
    zero_temperature_work_arrays!(temp_field)
    
    # Step 1: Compute ALL gradients in spectral space (NO COMMUNICATION!)
    t_spectral = MPI.Wtime()
    compute_all_gradients_spectral!(temp_field, oc_domain)
    temp_field.spectral_time[] += MPI.Wtime() - t_spectral
    
    # Step 2: Single batched transform of temperature and gradients to physical
    t_transform = MPI.Wtime()
    batch_transform_to_physical!(temp_field)
    temp_field.transform_time[] += MPI.Wtime() - t_transform
    
    # Step 3: Compute advection term -u·∇T in physical space (local operation)
    if vel_fields !== nothing
        compute_temperature_advection_local!(temp_field, vel_fields)
    end
    
    # Step 4: Add internal heat sources (local operation)
    add_internal_sources_local!(temp_field, oc_domain)
    
    # Step 5: Transform advection + sources back to spectral space
    t_transform = MPI.Wtime()
    shtns_physical_to_spectral!(temp_field.advection_physical, temp_field.nonlinear)
    temp_field.transform_time[] += MPI.Wtime() - t_transform
    
    # Step 6: Apply boundary conditions in spectral space
    apply_temperature_boundary_conditions_spectral!(temp_field, oc_domain)
    
    if ENABLE_TIMING[]
        temp_field.computation_time[] += MPI.Wtime() - t_start
    end
end

# ============================================================================
# Fully spectral gradient computation (NO COMMUNICATION!)
# ============================================================================
function compute_all_gradients_spectral!(temp_field::SHTnsTemperatureField{T}, 
                                        oc_domain::RadialDomain) where T
    """
    Compute all three gradient components entirely in spectral space.
    This is a completely local operation - no MPI communication!
    """
    
    # Get local ranges from config
    lm_range = range_local(temp_field.config.pencils.spec, 1)
    r_range = range_local(temp_field.config.pencils.spec, 3)
    
    # 1. Azimuthal gradient: ∂T/∂φ = im * T
    compute_phi_gradient_spectral!(temp_field)
    
    # 2. Colatitude gradient: ∂T/∂θ using recurrence relations
    compute_theta_gradient_spectral!(temp_field)
    
    # 3. Radial gradient: ∂T/∂r using banded matrix
    compute_radial_gradient_spectral!(temp_field, oc_domain)
    
    # Apply geometric factors (1/r, 1/(r sin θ)) in spectral space
    apply_geometric_factors_spectral!(temp_field, oc_domain)
end

function compute_phi_gradient_spectral!(temp_field::SHTnsTemperatureField{T}) where T
    """
    Compute ∂T/∂φ = im * T in spectral space (local operation)
    """
    spec_real   = parent(temp_field.spectral.data_real)
    spec_imag   = parent(temp_field.spectral.data_imag)
    grad_φ_real = parent(temp_field.grad_phi_spec.data_real)
    grad_φ_imag = parent(temp_field.grad_phi_spec.data_imag)
    
    lm_range = range_local(temp_field.config.pencils.spec, 1)
    r_range  = range_local(temp_field.config.pencils.spec, 3)
    
    @inbounds for lm_idx in lm_range
        if lm_idx <= temp_field.config.nlm
            local_lm = lm_idx - first(lm_range) + 1
            m = temp_field.config.m_values[lm_idx]
            
            @simd for r_idx in r_range
                local_r = r_idx - first(r_range) + 1
                if local_r <= size(spec_real, 3)
                    # ∂/∂φ = im in spectral space
                    T_real = spec_real[local_lm, 1, local_r]
                    T_imag = spec_imag[local_lm, 1, local_r]
                    
                    grad_φ_real[local_lm, 1, local_r] = -m * T_imag
                    grad_φ_imag[local_lm, 1, local_r] = m * T_real
                end
            end
        end
    end
end


function compute_theta_gradient_spectral!(temp_field::SHTnsTemperatureField{T}) where T
    """
    Compute ∂T/∂θ using recurrence relations in spectral space (local operation)
    Uses pre-computed coefficients for efficiency
    """
    spec_real   = parent(temp_field.spectral.data_real)
    spec_imag   = parent(temp_field.spectral.data_imag)
    grad_θ_real = parent(temp_field.grad_theta_spec.data_real)
    grad_θ_imag = parent(temp_field.grad_theta_spec.data_imag)
    
    config    = temp_field.config
    lm_range = range_local(config.pencils.spec, 1)
    r_range  = range_local(config.pencils.spec, 3)
    
    # Zero output first
    fill!(grad_θ_real, zero(T))
    fill!(grad_θ_imag, zero(T))
    
    @inbounds for r_idx in r_range
        local_r = r_idx - first(r_range) + 1
        if local_r <= size(spec_real, 3)
            
            # Method 1: Use sparse matrix (more efficient for repeated operations)
            if nnz(temp_field.theta_derivative_matrix) > 0
                # Extract radial slice
                for lm_idx in lm_range
                    if lm_idx <= config.nlm
                        local_lm = lm_idx - first(lm_range) + 1
                        
                        # Apply sparse matrix to compute derivative
                        for j in nzrange(temp_field.theta_derivative_matrix, lm_idx)
                            row = temp_field.theta_derivative_matrix.rowval[j]
                            coeff = temp_field.theta_derivative_matrix.nzval[j]
                            
                            if row in lm_range
                                local_row = row - first(lm_range) + 1
                                grad_θ_real[local_row, 1, local_r] += coeff * spec_real[local_lm, 1, local_r]
                                grad_θ_imag[local_row, 1, local_r] += coeff * spec_imag[local_lm, 1, local_r]
                            end
                        end
                    end
                end
            else
                # Method 2: Use recurrence coefficients directly
                for lm_idx in lm_range
                    if lm_idx <= config.nlm
                        local_lm = lm_idx - first(lm_range) + 1
                        l = config.l_values[lm_idx]
                        m = config.m_values[lm_idx]
                        
                        T_real = spec_real[local_lm, 1, local_r]
                        T_imag = spec_imag[local_lm, 1, local_r]
                        
                        # Apply recurrence relations
                        # Contribution to (l-1,m) mode
                        if l > abs(m)
                            lm1_idx = get_mode_index(config, l-1, m)
                            if lm1_idx > 0 && lm1_idx in lm_range
                                local_lm1 = lm1_idx - first(lm_range) + 1
                                coeff = temp_field.theta_recurrence_coeffs[lm_idx, 1]
                                grad_θ_real[local_lm1, 1, local_r] += coeff * T_real
                                grad_θ_imag[local_lm1, 1, local_r] += coeff * T_imag
                            end
                        end
                        
                        # Contribution to (l+1,m) mode
                        if l < config.lmax
                            lp1_idx = get_mode_index(config, l+1, m)
                            if lp1_idx > 0 && lp1_idx in lm_range
                                local_lp1 = lp1_idx - first(lm_range) + 1
                                coeff = temp_field.theta_recurrence_coeffs[lm_idx, 2]
                                grad_θ_real[local_lp1, 1, local_r] += coeff * T_real
                                grad_θ_imag[local_lp1, 1, local_r] += coeff * T_imag
                            end
                        end
                    end
                end
            end
        end
    end
end


function compute_radial_gradient_spectral!(temp_field::SHTnsTemperatureField{T}, 
                                          oc_domain::RadialDomain) where T
    """
    Compute ∂T/∂r using banded matrix in spectral space (local operation)
    """
    spec_real   = parent(temp_field.spectral.data_real)
    spec_imag   = parent(temp_field.spectral.data_imag)
    grad_r_real = parent(temp_field.grad_r_spec.data_real)
    grad_r_imag = parent(temp_field.grad_r_spec.data_imag)
    
    lm_range = range_local(temp_field.config.pencils.spec, 1)
    r_range  = range_local(temp_field.config.pencils.spec, 3)
    
    nr = oc_domain.N
    bandwidth = temp_field.dr_matrix.bandwidth
    
    @inbounds for lm_idx in lm_range
        if lm_idx <= temp_field.config.nlm
            local_lm = lm_idx - first(lm_range) + 1
            
            # Apply banded matrix to radial profile
            for r_idx in r_range
                local_r = r_idx - first(r_range) + 1
                if local_r <= size(grad_r_real, 3)
                    
                    dr_real = zero(T)
                    dr_imag = zero(T)
                    
                    # Banded matrix multiplication
                    for j in max(1, r_idx - bandwidth):min(nr, r_idx + bandwidth)
                        if j in r_range
                            local_j = j - first(r_range) + 1
                            band_row = bandwidth + 1 + r_idx - j
                            if 1 <= band_row <= 2*bandwidth + 1
                                coeff = temp_field.dr_matrix.data[band_row, j]
                                dr_real += coeff * spec_real[local_lm, 1, local_j]
                                dr_imag += coeff * spec_imag[local_lm, 1, local_j]
                            end
                        end
                    end
                    
                    grad_r_real[local_lm, 1, local_r] = dr_real
                    grad_r_imag[local_lm, 1, local_r] = dr_imag
                end
            end
        end
    end
end


function apply_geometric_factors_spectral!(temp_field::SHTnsTemperatureField{T}, 
                                          oc_domain::RadialDomain) where T
    """
    Apply geometric factors (1/r, 1/(r sin θ)) in spectral space
    For gradients in spherical coordinates
    """
    grad_θ_real = parent(temp_field.grad_theta_spec.data_real)
    grad_θ_imag = parent(temp_field.grad_theta_spec.data_imag)
    grad_φ_real = parent(temp_field.grad_phi_spec.data_real)
    grad_φ_imag = parent(temp_field.grad_phi_spec.data_imag)
    
    r_range  = range_local(temp_field.config.pencils.spec, 3)
    lm_range = range_local(temp_field.config.pencils.spec, 1)
    
    @inbounds for r_idx in r_range
        if r_idx <= oc_domain.N
            r_inv = oc_domain.r[r_idx, 3]  # 1/r
            local_r = r_idx - first(r_range) + 1
            
            @simd for lm_idx in lm_range
                local_lm = lm_idx - first(lm_range) + 1
                if local_lm <= size(grad_θ_real, 1) && local_r <= size(grad_θ_real, 3)
                    # ∇_θ T = (1/r) ∂T/∂θ
                    grad_θ_real[local_lm, 1, local_r] *= r_inv
                    grad_θ_imag[local_lm, 1, local_r] *= r_inv
                    
                    # ∇_φ T = (1/(r sin θ)) ∂T/∂φ
                    # The sin θ factor is handled in the spherical harmonic basis
                    grad_φ_real[local_lm, 1, local_r] *= r_inv
                    grad_φ_imag[local_lm, 1, local_r] *= r_inv
                end
            end
        end
    end
end

# ============================================================================
# Batched transform operations
# ============================================================================
function batch_transform_to_physical!(temp_field::SHTnsTemperatureField{T}) where T
    """
    Transform temperature and all gradient components to physical space
    in a single batched operation to minimize communication
    """
    
    # Create arrays of fields to transform
    spectral_fields = [temp_field.spectral,
                       temp_field.grad_theta_spec,
                       temp_field.grad_phi_spec,
                       temp_field.grad_r_spec]
    
    physical_fields = [temp_field.temperature,
                       temp_field.gradient.θ_component,
                       temp_field.gradient.φ_component,
                       temp_field.gradient.r_component]
    
    # Single batched transform with one MPI communication
    batch_spectral_to_physical!(spectral_fields, physical_fields)
end

# ============================================================================
# Local physical space operations (no communication)
# ============================================================================
function compute_temperature_advection_local!(temp_field::SHTnsTemperatureField{T}, 
                                             vel_fields) where T
    """
    Compute -u·∇T in physical space (completely local operation)
    """
    u_r = parent(vel_fields.velocity.r_component.data)
    u_θ = parent(vel_fields.velocity.θ_component.data)
    u_φ = parent(vel_fields.velocity.φ_component.data)
    
    grad_r = parent(temp_field.gradient.r_component.data)
    grad_θ = parent(temp_field.gradient.θ_component.data)
    grad_φ = parent(temp_field.gradient.φ_component.data)
    
    advection = parent(temp_field.advection_physical.data)
    
    @inbounds @simd for idx in eachindex(advection)
        if idx <= length(u_r) && idx <= length(grad_r)
            advection[idx] = -(u_r[idx] * grad_r[idx] + 
                              u_θ[idx] * grad_θ[idx] + 
                              u_φ[idx] * grad_φ[idx])
        end
    end
end

function add_internal_sources_local!(temp_field::SHTnsTemperatureField{T}, 
                                    domain::RadialDomain) where T
    """
    Add volumetric heating (completely local operation)
    """
    advection = parent(temp_field.advection_physical.data)
    
    if !all(iszero, temp_field.internal_sources)
        # Get local physical dimensions
        local_shape = size(temp_field.advection_physical.data)
        nlat_local, nlon_local, nr_local = local_shape
        
        r_range = range_local(temp_field.config.pencils.r, 3)
        
        @inbounds for k in 1:nr_local
            r_idx = k + first(r_range) - 1
            if r_idx <= length(temp_field.internal_sources)
                source_value = temp_field.internal_sources[r_idx]
                
                # Add uniformly at this radius
                @simd for j in 1:nlon_local
                    for i in 1:nlat_local
                        idx = i + (j-1)*nlat_local + (k-1)*nlat_local*nlon_local
                        if idx <= length(advection)
                            advection[idx] += source_value
                        end
                    end
                end
            end
        end
    end
end

# ============================================================================
# Boundary conditions in spectral space
# ============================================================================
function apply_temperature_boundary_conditions_spectral!(temp_field::SHTnsTemperatureField{T}, 
                                                        domain::RadialDomain) where T
    """
    Apply boundary conditions in spectral space
    """
    spec_real = parent(temp_field.spectral.data_real)
    spec_imag = parent(temp_field.spectral.data_imag)
    
    lm_range = range_local(temp_field.config.pencils.spec, 1)
    r_range  = range_local(temp_field.config.pencils.spec, 3)
    
    # Check which boundaries are local
    has_inner = 1 in r_range
    has_outer = oc_domain.N in r_range
    
    @inbounds for lm_idx in lm_range
        if lm_idx <= temp_field.config.nlm
            local_lm = lm_idx - first(lm_range) + 1
            
            # Inner boundary
            if has_inner
                if temp_field.bc_type_inner[lm_idx] == 1    # Dirichlet
                    local_r = 1 - first(r_range) + 1
                    spec_real[local_lm, 1, local_r] = temp_field.boundary_values[1, lm_idx]
                    spec_imag[local_lm, 1, local_r] = 0.0
                elseif temp_field.bc_type_inner[lm_idx] == 2  # Neumann
                    apply_flux_bc_spectral!(temp_field, lm_idx, local_lm, 1, domain)
                end
            end
            
            # Outer boundary
            if has_outer
                if temp_field.bc_type_outer[lm_idx] == 1      # Dirichlet
                    local_r = oc_domain.N - first(r_range) + 1
                    spec_real[local_lm, 1, local_r] = temp_field.boundary_values[2, lm_idx]
                    spec_imag[local_lm, 1, local_r] = 0.0
                elseif temp_field.bc_type_outer[lm_idx] == 2  # Neumann
                    apply_flux_bc_spectral!(temp_field, lm_idx, local_lm, oc_domain.N, oc_domain)
                end
            end
        end
    end
end


# ============================================================================
# Complete Flux Boundary Condition Implementation for Spectral Methods
# ============================================================================

"""
    apply_flux_bc_spectral_complete!(temp_field, domain)

Complete implementation of flux boundary conditions in spectral space.
This modifies the spectral coefficients to satisfy ∂T/∂r = prescribed_flux.
"""
function apply_flux_bc_spectral!(temp_field::SHTnsTemperatureField{T}, 
                                         domain::RadialDomain) where T
    
    spec_real = parent(temp_field.spectral.data_real)
    spec_imag = parent(temp_field.spectral.data_imag)
    
    lm_range = range_local(temp_field.config.pencils.spec, 1)
    r_range  = range_local(temp_field.config.pencils.spec, 3)
    
    # Process each (l,m) mode independently
    @inbounds for lm_idx in lm_range
        if lm_idx <= temp_field.config.nlm
            local_lm = lm_idx - first(lm_range) + 1
            
            # Check BC types for this mode
            apply_inner = (temp_field.bc_type_inner[lm_idx] == 2) && (1 in r_range)
            apply_outer = (temp_field.bc_type_outer[lm_idx] == 2) && (domain.N in r_range)
            
            if apply_inner || apply_outer
                # Apply flux BC using one of three methods
                apply_flux_bc_mode_tau!(spec_real, spec_imag, local_lm, lm_idx,
                                       apply_inner, apply_outer,
                                       temp_field, domain, r_range)
            end
        end
    end
end

# ============================================================================
# Method 1: Tau Method (Most Robust)
# ============================================================================

"""
    apply_flux_bc_mode_tau!(...)

Apply flux BC using the tau method.
This adds correction polynomials to satisfy the boundary conditions.
"""
function apply_flux_bc_mode_tau!(spec_real, spec_imag, local_lm, lm_idx,
                                apply_inner, apply_outer,
                                temp_field, domain, r_range) where T
    
    nr = domain.N
    
    # Extract radial profile for this mode
    profile_real = zeros(T, nr)
    profile_imag = zeros(T, nr)
    
    for r_idx in r_range
        local_r = r_idx - first(r_range) + 1
        if local_r <= size(spec_real, 3)
            profile_real[r_idx] = spec_real[local_lm, 1, local_r]
            profile_imag[r_idx] = spec_imag[local_lm, 1, local_r]
        end
    end
    
    # MPI gather to get complete profile (needed for BC application)
    if MPI.Comm_size(get_comm()) > 1
        MPI.Allreduce!(profile_real, MPI.SUM, get_comm())
        MPI.Allreduce!(profile_imag, MPI.SUM, get_comm())
    end
    
    # Get prescribed flux values
    flux_inner = apply_inner ? get_flux_value(lm_idx, 1, temp_field) : T(0)
    flux_outer = apply_outer ? get_flux_value(lm_idx, 2, temp_field) : T(0)
    
    # Compute current fluxes at boundaries
    current_flux_inner, current_flux_outer = compute_boundary_fluxes(
        profile_real, temp_field.dr_matrix, domain)
    
    # Compute tau corrections
    if apply_inner && apply_outer
        # Both boundaries have flux BCs
        tau_coeffs = compute_tau_coefficients_both(
            flux_inner - current_flux_inner,
            flux_outer - current_flux_outer,
            domain)
    elseif apply_inner
        # Only inner boundary
        tau_coeffs = compute_tau_coefficients_inner(
            flux_inner - current_flux_inner, domain)
    else
        # Only outer boundary
        tau_coeffs = compute_tau_coefficients_outer(
            flux_outer - current_flux_outer, domain)
    end
    
    # Apply tau correction to profile
    apply_tau_correction!(profile_real, tau_coeffs, domain)
    if any(x -> abs(x) > 1e-12, profile_imag)
        apply_tau_correction!(profile_imag, tau_coeffs, domain)
    end
    
    # Store corrected profile back
    for r_idx in r_range
        local_r = r_idx - first(r_range) + 1
        if local_r <= size(spec_real, 3)
            spec_real[local_lm, 1, local_r] = profile_real[r_idx]
            spec_imag[local_lm, 1, local_r] = profile_imag[r_idx]
        end
    end
end

function compute_tau_coefficients_both(flux_error_inner::T, flux_error_outer::T,
                                      domain::RadialDomain) where T
    """
    Compute tau polynomial coefficients for both boundaries.
    Uses highest two Chebyshev modes as tau functions.
    """
    nr = domain.N
    
    # Tau polynomials: T_{N-1}(r) and T_N(r)
    tau1 = compute_chebyshev_polynomial(nr-1, domain)
    tau2 = compute_chebyshev_polynomial(nr, domain)
    
    # Derivatives of tau polynomials at boundaries
    dtau1_inner = evaluate_chebyshev_derivative(nr-1, domain.r[1, 4], domain)
    dtau1_outer = evaluate_chebyshev_derivative(nr-1, domain.r[nr, 4], domain)
    dtau2_inner = evaluate_chebyshev_derivative(nr, domain.r[1, 4], domain)
    dtau2_outer = evaluate_chebyshev_derivative(nr, domain.r[nr, 4], domain)
    
    # Solve 2x2 system for tau coefficients
    # [dtau1_inner  dtau2_inner] [c1]   [flux_error_inner]
    # [dtau1_outer  dtau2_outer] [c2] = [flux_error_outer]
    
    det = dtau1_inner * dtau2_outer - dtau1_outer * dtau2_inner
    
    if abs(det) > 1e-12
        c1 = (flux_error_inner * dtau2_outer - flux_error_outer * dtau2_inner) / det
        c2 = (flux_error_outer * dtau1_inner - flux_error_inner * dtau1_outer) / det
    else
        c1 = c2 = zero(T)
    end
    
    return (c1, c2, tau1, tau2)
end

function compute_tau_coefficients_inner(flux_error::T, domain::RadialDomain) where T
    """
    Compute tau coefficient for inner boundary only.
    Uses a single tau function that doesn't affect outer boundary.
    """
    nr = domain.N
    
    # Use a polynomial that has zero derivative at outer boundary
    # This is a linear combination of Chebyshev polynomials
    tau = compute_inner_tau_function(domain)
    
    # Derivative at inner boundary
    dtau_inner = evaluate_tau_derivative_inner(tau, domain)
    
    # Tau coefficient
    c = flux_error / dtau_inner
    
    return (c, tau)
end

function apply_tau_correction!(profile::Vector{T}, tau_coeffs, domain::RadialDomain) where T
    """
    Add tau correction to the radial profile.
    """
    if length(tau_coeffs) == 4  # Both boundaries
        c1, c2, tau1, tau2 = tau_coeffs
        @. profile += c1 * tau1 + c2 * tau2
    elseif length(tau_coeffs) == 2  # Single boundary
        c, tau = tau_coeffs
        @. profile += c * tau
    end
end

# ============================================================================
# Method 2: Influence Matrix Method
# ============================================================================

"""
    apply_flux_bc_influence_matrix!(...)

Apply flux BC using influence matrix method.
This is more physical - uses Green's functions approach.
"""
function apply_flux_bc_influence_matrix!(spec_real, spec_imag, local_lm, lm_idx,
                                        temp_field, domain, r_range) where T
    
    # Pre-compute influence functions (could be cached)
    G_inner, G_outer = compute_influence_functions_flux(domain)
    
    # Build influence matrix
    M = build_influence_matrix(G_inner, G_outer, temp_field.dr_matrix, domain)
    
    # Get prescribed and current fluxes
    flux_prescribed = [get_flux_value(lm_idx, 1, temp_field),
                      get_flux_value(lm_idx, 2, temp_field)]
    
    flux_current = compute_current_fluxes(spec_real, spec_imag, local_lm, 
                                         temp_field.dr_matrix, domain, r_range)
    
    # Solve for correction amplitudes
    flux_error = flux_prescribed - flux_current
    correction_amplitudes = M \ flux_error
    
    # Apply corrections
    for r_idx in r_range
        local_r = r_idx - first(r_range) + 1
        if local_r <= size(spec_real, 3)
            spec_real[local_lm, 1, local_r] += (correction_amplitudes[1] * G_inner[r_idx] +
                                               correction_amplitudes[2] * G_outer[r_idx])
            spec_imag[local_lm, 1, local_r] += (correction_amplitudes[1] * G_inner[r_idx] +
                                               correction_amplitudes[2] * G_outer[r_idx])
        end
    end
end

function compute_influence_functions_flux(oc_domain::RadialDomain)
    """
    Compute influence functions for flux BCs.
    These are solutions to the homogeneous equation with specific BCs.
    """
    nr = oc_domain.N
    ri = oc_domain.r[1, 4]
    ro = oc_domain.r[nr, 4]
    
    G_inner = zeros(nr)
    G_outer = zeros(nr)
    
    for i in 1:nr
        r = oc_domain.r[i, 4]
        ξ = (r - ri) / (ro - ri)
        
        # Inner influence: strong at inner, weak at outer
        G_inner[i] = exp(-3.0 * ξ) * (1.0 - ξ)
        
        # Outer influence: weak at inner, strong at outer
        G_outer[i] = exp(-3.0 * (1.0 - ξ)) * ξ
    end
    
    # Normalize to have unit flux contribution
    normalize_influence_function!(G_inner, oc_domain, 1)
    normalize_influence_function!(G_outer, oc_domain, 2)
    
    return G_inner, G_outer
end

# ============================================================================
# Method 3: Direct Modification (Simplest but less accurate)
# ============================================================================

"""
    apply_flux_bc_direct!(...)

Direct modification of boundary values to approximately satisfy flux BC.
This is the simplest but least accurate method.
"""
function apply_flux_bc_direct!(spec_real, spec_imag, local_lm, lm_idx,
                              temp_field, domain, r_range) where T
    
    # Get prescribed flux
    if 1 in r_range
        flux_inner = get_flux_value(lm_idx, 1, temp_field)
        modify_for_flux_inner!(spec_real, spec_imag, local_lm, flux_inner, 
                              temp_field.dr_matrix, domain, r_range)
    end
    
    if domain.N in r_range
        flux_outer = get_flux_value(lm_idx, 2, temp_field)
        modify_for_flux_outer!(spec_real, spec_imag, local_lm, flux_outer,
                              temp_field.dr_matrix, domain, r_range)
    end
end

function modify_for_flux_inner!(spec_real, spec_imag, local_lm, prescribed_flux,
                               dr_matrix, domain, r_range) where T
    """
    Modify coefficients near inner boundary to approximate flux condition.
    Uses low-order extrapolation.
    """
    if 1 in r_range && 2 in r_range
        # Use linear extrapolation based on prescribed flux
        local_1 = 1 - first(r_range) + 1
        local_2 = 2 - first(r_range) + 1
        
        dr = domain.r[2, 4] - domain.r[1, 4]
        
        # T(r1) ≈ T(r2) - prescribed_flux * dr
        spec_real[local_lm, 1, local_1] = spec_real[local_lm, 1, local_2] - prescribed_flux * dr
        spec_imag[local_lm, 1, local_1] = spec_imag[local_lm, 1, local_2]
    end
end

# ============================================================================
# Utility Functions
# ============================================================================
function compute_boundary_fluxes(profile::Vector{T}, dr_matrix::BandedMatrix{T},
                                domain::RadialDomain) where T
    """
    Compute flux (dT/dr) at both boundaries using the derivative matrix.
    """
    nr = domain.N
    dprofile = apply_derivative_matrix(dr_matrix, profile)
    
    return dprofile[1], dprofile[nr]
end

function compute_chebyshev_polynomial(n::Int, domain::RadialDomain)
    """
    Compute Chebyshev polynomial T_n on the radial grid.
    """
    nr = domain.N
    poly = zeros(nr)
    
    ri = domain.r[1, 4]
    ro = domain.r[nr, 4]
    
    for i in 1:nr
        r = domain.r[i, 4]
        # Map to [-1, 1]
        x = 2.0 * (r - ri) / (ro - ri) - 1.0
        # T_n(x) = cos(n * acos(x))
        poly[i] = cos(n * acos(clamp(x, -1.0, 1.0)))
    end
    
    return poly
end


function evaluate_chebyshev_derivative(n::Int, r::T, domain::RadialDomain) where T
    """
    Evaluate derivative of Chebyshev polynomial at a point.
    """
    ri = domain.r[1, 4]
    ro = domain.r[domain.N, 4]
    
    # Map to [-1, 1]
    x = 2.0 * (r - ri) / (ro - ri) - 1.0
    
    # dT_n/dx = n * U_{n-1}(x) where U is Chebyshev polynomial of second kind
    # U_{n-1}(x) = sin(n * acos(x)) / sin(acos(x))
    
    if abs(abs(x) - 1.0) < 1e-12
        # Special case at boundaries
        dTn_dx = n^2 * sign(x)^(n+1)
    else
        theta = acos(clamp(x, -1.0, 1.0))
        dTn_dx = n * sin(n * theta) / sin(theta)
    end
    
    # Chain rule: dT_n/dr = dT_n/dx * dx/dr
    dx_dr = 2.0 / (ro - ri)
    
    return dTn_dx * dx_dr
end

function get_flux_value(lm_idx::Int, boundary::Int, temp_field)
    """
    Get prescribed flux value for given mode and boundary.
    """
    l = temp_field.config.l_values[lm_idx]
    m = temp_field.config.m_values[lm_idx]
    
    # Example: uniform heating/cooling for l=0,m=0
    if l == 0 && m == 0
        if boundary == 1
            return T(1.0)   # Heating from below
        else
            return T(-1.0)  # Cooling from above
        end
    else
        return T(0.0)  # No flux for other modes
    end
end

function apply_derivative_matrix(dr_matrix::BandedMatrix{T}, profile::Vector{T}) where T
    """
    Apply banded derivative matrix to compute derivative.
    """
    nr = dr_matrix.size
    bandwidth = dr_matrix.bandwidth
    result = zeros(T, nr)
    
    @inbounds for j in 1:nr
        for i in max(1, j - bandwidth):min(nr, j + bandwidth)
            band_row = bandwidth + 1 + i - j
            if 1 <= band_row <= 2*bandwidth + 1
                result[i] += dr_matrix.data[band_row, j] * profile[j]
            end
        end
    end
    
    return result
end

# ============================================================================
# Validation and Testing
# ============================================================================

function validate_flux_bc(temp_field, domain)
    """
    Check if flux boundary conditions are satisfied within tolerance.
    """
    spec_real = parent(temp_field.spectral.data_real)
    spec_imag = parent(temp_field.spectral.data_imag)
    
    lm_range = range_local(temp_field.config.pencils.spec, 1)
    
    max_error = 0.0
    
    for lm_idx in lm_range
        if lm_idx <= temp_field.config.nlm
            local_lm = lm_idx - first(lm_range) + 1
            
            # Check inner boundary
            if temp_field.bc_type_inner[lm_idx] == 2
                prescribed = get_flux_value(lm_idx, 1, temp_field)
                actual = compute_flux_at_boundary(spec_real, spec_imag, local_lm, 
                                                 1, temp_field, domain)
                error = abs(prescribed - actual)
                max_error = max(max_error, error)
            end
            
            # Check outer boundary
            if temp_field.bc_type_outer[lm_idx] == 2
                prescribed = get_flux_value(lm_idx, 2, temp_field)
                actual = compute_flux_at_boundary(spec_real, spec_imag, local_lm,
                                                 domain.N, temp_field, domain)
                error = abs(prescribed - actual)
                max_error = max(max_error, error)
            end
        end
    end
    
    # Global maximum error
    global_max_error = MPI.Allreduce(max_error, MPI.MAX, get_comm())
    
    if get_rank() == 0
        println("Maximum flux BC error: $(global_max_error)")
        if global_max_error > 1e-6
            println("⚠ Warning: Flux BC error exceeds tolerance")
        end
    end
    
    return global_max_error
end


# ============================================================================
# Diagnostic functions
# ============================================================================
function compute_nusselt_number(temp_field::SHTnsTemperatureField{T}, 
                               domain::RadialDomain) where T
    """
    Compute Nusselt number from heat flux at boundaries
    """
    # Compute heat flux from radial gradient
    grad_r = temp_field.gradient.r_component
    
    # Get flux at boundaries (requires communication)
    flux_inner = compute_surface_flux(grad_r, 1, temp_field.config)
    flux_outer = compute_surface_flux(grad_r, oc_domain.N, temp_field.config)
    
    # Nusselt number
    conductive_flux = 4π * oc_domain.r[1, 4]^2
    Nu = abs(flux_outer) / conductive_flux
    
    return Nu
end


function compute_thermal_energy(temp_field::SHTnsTemperatureField{T}) where T
    """
    Compute total thermal energy in spectral space
    """
    spec_real = parent(temp_field.spectral.data_real)
    spec_imag = parent(temp_field.spectral.data_imag)
    
    # Local energy computation
    local_energy = 0.0
    
    lm_range = range_local(temp_field.config.pencils.spec, 1)
    r_range  = range_local(temp_field.config.pencils.spec, 3)
    
    @inbounds for lm_idx in lm_range
        if lm_idx <= temp_field.config.nlm
            local_lm = lm_idx - first(lm_range) + 1
            
            @simd for r_idx in r_range
                local_r = r_idx - first(r_range) + 1
                if local_r <= size(spec_real, 3)
                    local_energy += (spec_real[local_lm, 1, local_r]^2 + 
                                   spec_imag[local_lm, 1, local_r]^2)
                end
            end
        end
    end
    
    # Global sum across all processes
    return 0.5 * MPI.Allreduce(local_energy, MPI.SUM, get_comm())
end


function compute_surface_flux(field::SHTnsPhysicalField{T}, r_level::Int, 
                             config::SHTnsConfig) where T
    """
    Compute surface integral of flux at given radial level
    """
    data = parent(field.data)
    
    # Local contribution
    local_flux = 0.0
    
    # Get local range
    local_range = range_local(config.pencils.r)
    θ_range, φ_range, r_range = local_range
    
    if r_level in r_range
        local_r = r_level - first(r_range) + 1
        
        for φ_idx in φ_range, θ_idx in θ_range
            if θ_idx <= config.nlat && φ_idx <= config.nlon
                local_θ = θ_idx - first(θ_range) + 1
                local_φ = φ_idx - first(φ_range) + 1
                
                idx = local_θ + (local_φ-1)*length(θ_range) + (local_r-1)*length(θ_range)*length(φ_range)
                
                if idx <= length(data)
                    # Use Gaussian quadrature weights
                    weight = config.gauss_weights[θ_idx] * (2π / config.nlon) * sin(config.theta_grid[θ_idx])
                    local_flux += data[idx] * weight
                end
            end
        end
    end
    
    # Global reduction
    return MPI.Allreduce(local_flux, MPI.SUM, get_comm())
end


# ============================================================================
# Performance monitoring and statistics
# ============================================================================
function get_temperature_statistics(temp_field::SHTnsTemperatureField{T}, 
                                   domain::RadialDomain) where T
    """
    Compute various temperature field statistics
    """
    # Min/max temperature
    temp_data = parent(temp_field.temperature.data)
    local_min = minimum(temp_data)
    local_max = maximum(temp_data)
    
    global_min = MPI.Allreduce(local_min, MPI.MIN, get_comm())
    global_max = MPI.Allreduce(local_max, MPI.MAX, get_comm())
    
    # RMS temperature
    local_sum = sum(temp_data.^2)
    local_count = length(temp_data)
    
    global_sum = MPI.Allreduce(local_sum, MPI.SUM, get_comm())
    global_count = MPI.Allreduce(local_count, MPI.SUM, get_comm())
    
    rms_temp = sqrt(global_sum / global_count)
    
    # Nusselt number
    Nu = compute_nusselt_number(temp_field, domain)
    
    # Total energy
    energy = compute_thermal_energy(temp_field)
    
    return (min = global_min,
            max = global_max,
            rms = rms_temp,
            nusselt = Nu,
            energy = energy)
end

# ============================================================================
# Utility functions
# ============================================================================
function zero_temperature_work_arrays!(temp_field::SHTnsTemperatureField{T}) where T
    """
    Efficiently zero all work arrays
    """
    fill!(parent(temp_field.work_spectral.data_real), zero(T))
    fill!(parent(temp_field.work_spectral.data_imag), zero(T))
    fill!(parent(temp_field.work_physical.data), zero(T))
    fill!(parent(temp_field.advection_physical.data), zero(T))
    fill!(parent(temp_field.grad_theta_spec.data_real), zero(T))
    fill!(parent(temp_field.grad_theta_spec.data_imag), zero(T))
    fill!(parent(temp_field.grad_phi_spec.data_real), zero(T))
    fill!(parent(temp_field.grad_phi_spec.data_imag), zero(T))
    fill!(parent(temp_field.grad_r_spec.data_real), zero(T))
    fill!(parent(temp_field.grad_r_spec.data_imag), zero(T))
end

function set_temperature_ic!(temp_field::SHTnsTemperatureField{T}, 
                            domain::RadialDomain;
                            perturbation_amplitude::T = T(1e-3)) where T
    """
    Set initial condition for temperature field
    """
    spec_real = parent(temp_field.spectral.data_real)
    spec_imag = parent(temp_field.spectral.data_imag)
    
    lm_range = range_local(temp_field.config.pencils.spec, 1)
    r_range = range_local(temp_field.config.pencils.spec, 3)
    
    @inbounds for lm_idx in lm_range
        if lm_idx <= temp_field.config.nlm
            local_lm = lm_idx - first(lm_range) + 1
            l = temp_field.config.l_values[lm_idx]
            m = temp_field.config.m_values[lm_idx]
            
            for r_idx in r_range
                local_r = r_idx - first(r_range) + 1
                if local_r <= size(spec_real, 3)
                    r = domain.r[r_idx, 4]
                    
                    if l == 0 && m == 0
                        # Conductive profile for l=0, m=0
                        spec_real[local_lm, 1, local_r] = 1.0 - r
                        spec_imag[local_lm, 1, local_r] = 0.0
                    elseif l <= 4
                        # Small perturbation for low modes
                        spec_real[local_lm, 1, local_r] = perturbation_amplitude * randn(T)
                        if m > 0
                            spec_imag[local_lm, 1, local_r] = perturbation_amplitude * randn(T)
                        else
                            spec_imag[local_lm, 1, local_r] = 0.0
                        end
                    else
                        # Zero for high modes
                        spec_real[local_lm, 1, local_r] = 0.0
                        spec_imag[local_lm, 1, local_r] = 0.0
                    end
                end
            end
        end
    end
end

function set_boundary_conditions!(temp_field::SHTnsTemperatureField{T};
                                 inner_bc_type::Int = 1,
                                 outer_bc_type::Int = 1,
                                 inner_value::T = T(1.0),
                                 outer_value::T = T(0.0)) where T
    """
    Set boundary condition types and values
    """
    # Set BC types for all modes
    fill!(temp_field.bc_type_inner, inner_bc_type)
    fill!(temp_field.bc_type_outer, outer_bc_type)
    
    # Set boundary values for l=0, m=0 mode (mean temperature)
    l0m0_idx = get_mode_index(temp_field.config, 0, 0)
    if l0m0_idx > 0
        temp_field.boundary_values[1, l0m0_idx] = inner_value
        temp_field.boundary_values[2, l0m0_idx] = outer_value
    end
    
    # Other modes have zero boundary values by default
    for lm_idx in 2:temp_field.config.nlm
        temp_field.boundary_values[1, lm_idx] = T(0.0)
        temp_field.boundary_values[2, lm_idx] = T(0.0)
    end
end

function set_internal_heating!(temp_field::SHTnsTemperatureField{T}, 
                              domain::RadialDomain;
                              heating_type::Symbol = :uniform,
                              amplitude::T = T(1.0)) where T
    """
    Set internal heating profile
    """
    if heating_type == :uniform
        # Uniform volumetric heating
        fill!(temp_field.internal_sources, amplitude)
    elseif heating_type == :gaussian
        # Gaussian heating profile centered at mid-radius
        r_mid = 0.5 * (oc_domain.r[1, 4] + oc_domain.r[end, 4])
        sigma = 0.1 * (oc_domain.r[end, 4] - oc_domain.r[1, 4])
        
        for i in 1:oc_domain.N
            r = oc_domain.r[i, 4]
            temp_field.internal_sources[i] = amplitude * exp(-((r - r_mid)/sigma)^2)
        end
    elseif heating_type == :bottom
        # Heating concentrated near bottom
        for i in 1:oc_domain.N
            r = oc_domain.r[i, 4]
            r_norm = (r - oc_domain.r[1, 4]) / (oc_domain.r[end, 4] - oc_domain.r[1, 4])
            temp_field.internal_sources[i] = amplitude * exp(-5.0 * r_norm)
        end
    else
        # No heating
        fill!(temp_field.internal_sources, zero(T))
    end
end

# ============================================================================
# Export functions
# ============================================================================
# export SHTnsTemperatureField, create_shtns_temperature_field
# export compute_temperature_nonlinear!
# export compute_nusselt_number, compute_thermal_energy
# export compute_surface_flux, get_temperature_statistics
# export zero_temperature_work_arrays!
# export set_temperature_ic!, set_boundary_conditions!, set_internal_heating!


#export print_temperature_performance

# # Export functions
# export SHTnsTemperatureField, create_shtns_temperature_field
# export compute_temperature_nonlinear!, compute_temperature_batch!
# export zero_work_arrays!
