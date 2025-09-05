# ============================================================================
# Timestepping Module with SHTns
# ============================================================================

using MPI
using LinearAlgebra
    
# Timestepping state
mutable struct TimestepState
    time::Float64
    dt::Float64
    step::Int
    iteration::Int
    error::Float64
    converged::Bool
end

# ================================
# Exponential AB2 (EAB2) Utilities
# ================================

struct ETDCache{T}
    dt::Float64
    l_values::Vector{Int}
    E::Vector{Matrix{T}}      # exp(dt A_l) per l
    phi1::Vector{Matrix{T}}   # phi1(dt A_l) per l
end

"""
    create_etd_cache(config, domain, diffusivity, dt) -> ETDCache

Build per-l exponential cache for the linear operator A_l = diffusivity*(d²/dr² + (2/r)d/dr − l(l+1)/r²).
Computes exp(dt A_l) and phi1(dt A_l) via dense methods. Single-rank recommended.
"""
function create_etd_cache(::Type{T}, config::SHTnsKitConfig, domain::RadialDomain,
                          diffusivity::Float64, dt::Float64) where T
    lap = create_radial_laplacian(domain)
    nr = domain.N
    r_inv2 = @views domain.r[1:nr, 2]
    lvals = unique(config.l_values)
    E = Matrix{T}[]
    PHI1 = Matrix{T}[]
    for l in lvals
        # Build banded for A = ν*(d² + (2/r)d − l(l+1)/r²)
        Adata = diffusivity .* lap.data
        # Convert to dense and subtract l(l+1)/r² on diagonal
        Adense = banded_to_dense(BandedMatrix(Adata, i_KL, nr))
        lfac = Float64(l * (l + 1))
        @inbounds for n in 1:nr
            Adense[n, n] -= diffusivity * lfac * r_inv2[n]
        end
        # exp(dt A)
        Adt = dt .* Adense
        E_l = exp(Adt)
        push!(E, Matrix{T}(E_l))
        # phi1(dt A) = A^{-1} * (exp(dt A) − I) / dt
        F = (E_l - I) / dt
        fac = lu(Adense)
        phi1_l = fac \ F
        push!(PHI1, Matrix{T}(phi1_l))
    end
    return ETDCache{T}(dt, lvals, E, PHI1)
end

"""
    eab2_update!(u, nl, nl_prev, etd, config)

Apply EAB2 update per (l,m): u^{n+1} = E u^n + dt*phi1*(3/2 nl^n − 1/2 nl^{n−1}).
"""
function eab2_update!(u::SHTnsSpectralField{T}, nl::SHTnsSpectralField{T},
                      nl_prev::SHTnsSpectralField{T}, etd::ETDCache{T}, config::SHTnsKitConfig,
                      dt::Float64) where T
    u_real = parent(u.data_real); u_imag = parent(u.data_imag)
    n_real = parent(nl.data_real); n_imag = parent(nl.data_imag)
    p_real = parent(nl_prev.data_real); p_imag = parent(nl_prev.data_imag)
    lm_range = get_local_range(u.pencil, 1)
    r_range  = get_local_range(u.pencil, 3)
    # Build map from lm_idx to l index in etd
    for lm_idx in lm_range
        if lm_idx <= u.nlm
            l = config.l_values[lm_idx]
            lpos = findfirst(==(l), etd.l_values)
            E = etd.E[lpos]
            P1 = etd.phi1[lpos]
            ll = lm_idx - first(lm_range) + 1
            # Extract full radial vectors (assume single-rank or full local r)
            ur = Vector{T}(undef, size(u_real, 3))
            ui = Vector{T}(undef, size(u_imag, 3))
            nr_loc = length(ur)
            @inbounds for k in 1:nr_loc
                ur[k] = u_real[ll, 1, k]
                ui[k] = u_imag[ll, 1, k]
            end
            # Nonlinear combo
            nrn = Vector{T}(undef, nr_loc)
            nin = Vector{T}(undef, nr_loc)
            @inbounds for k in 1:nr_loc
                nrn[k] = (3/2)*n_real[ll,1,k] - (1/2)*p_real[ll,1,k]
                nin[k] = (3/2)*n_imag[ll,1,k] - (1/2)*p_imag[ll,1,k]
            end
            ur_new = E*ur + dt*(P1*nrn)
            ui_new = E*ui + dt*(P1*nin)
            @inbounds for k in 1:nr_loc
                u_real[ll,1,k] = ur_new[k]
                u_imag[ll,1,k] = ui_new[k]
            end
        end
    end
    return u
end
# Implicit matrices for each spherical harmonic mode (SHTns version)
struct SHTnsImplicitMatrices{T}
    matrices::Vector{BandedMatrix{T}}  # One for each l value
    factorizations::Vector{Any}       # LU factorizations
    l_values::Vector{Int}             # l values for indexing
end

function create_shtns_timestepping_matrices(config::SHTnsKitConfig, 
                                            domain::RadialDomain, 
                                            diffusivity::Float64, 
                                            dt::Float64)
    matrices = BandedMatrix{Float64}[]
    factorizations = Any[]
    l_values = Int[]

    # Precompute l set and invariants
    unique_l = unique(config.l_values)
    laplacian = create_radial_laplacian(domain)              # invariant across l
    r_inv_sq = @views domain.r[1:domain.N, 2]                # 1/r^2 per radius

    # Build a base banded operator: (1/dt) - diffusivity*d_implicit*(d²/dr² + (2/r)d/dr)
    base_banded = copy(laplacian.data)
    base_banded[i_KL + 1, :] .+= 1.0 / dt
    base_banded .*= -diffusivity * d_implicit

    base_matrix = BandedMatrix(base_banded, i_KL, domain.N)

    for l in unique_l
        push!(l_values, l)
        # Create a banded work copy
        work_banded = copy(base_banded)
        # Add spherical-harmonic term on the diagonal (banded center row)
        l_factor = Float64(l * (l + 1))
        @inbounds for n in 1:domain.N
            work_banded[i_KL + 1, n] += diffusivity * d_implicit * l_factor * r_inv_sq[n]
        end

        # Store descriptor (for completeness)
        push!(matrices, base_matrix)

        # Factorize banded operator
        factorization = factorize_banded(BandedMatrix(work_banded, i_KL, domain.N))
        push!(factorizations, factorization)
    end

    return SHTnsImplicitMatrices(matrices, factorizations, l_values)
end

function banded_to_dense(matrix::BandedMatrix{T}) where T
    # Convert banded matrix to dense for LU factorization
    N = matrix.size
    bandwidth = matrix.bandwidth
    dense = zeros(T, N, N)
    
    for j in 1:N
        for i in max(1, j - bandwidth):min(N, j + bandwidth)
            band_row = bandwidth + 1 + i - j
            dense[i, j] = matrix.data[band_row, j]
        end
    end
    
    return dense
end

function apply_explicit_operator!(output::SHTnsSpectralField{T}, 
                                input::SHTnsSpectralField{T},
                                nonlinear::SHTnsSpectralField{T}, 
                                domain::RadialDomain,
                                diffusivity::Float64, 
                                dt::Float64) where T

    # Explicit operator: 
    # (1/dt + diffusivity * (1-implicit_factor) * L_l) * input + nonlinear
    
# Get local data
    out_real = parent(output.data_real)
    out_imag = parent(output.data_imag)
    in_real  = parent(input.data_real)
    in_imag  = parent(input.data_imag)
    nl_real  = parent(nonlinear.data_real)
    nl_imag  = parent(nonlinear.data_imag)
    
    # Get local ranges
    lm_range = get_local_range(input.pencil, 1)
    r_range  = get_local_range(input.pencil, 3)
    
    for lm_idx in lm_range
        if lm_idx <= input.nlm
            l = input.config.l_values[lm_idx]
            local_lm = lm_idx - first(lm_range) + 1
            
            for r_idx in r_range
                local_r = r_idx - first(r_range) + 1
                
                if local_lm <= size(out_real, 1) && local_r <= size(out_real, 3)
                    # Time derivative
                    out_real[local_lm, 1, local_r] = in_real[local_lm, 1, local_r] / dt
                    out_imag[local_lm, 1, local_r] = in_imag[local_lm, 1, local_r] / dt
                    
                    # Add nonlinear terms
                    out_real[local_lm, 1, local_r] += nl_real[local_lm, 1, local_r]
                    out_imag[local_lm, 1, local_r] += nl_imag[local_lm, 1, local_r]
                end
            end
        end
    end
end

"""
    build_rhs_cnab2!(rhs, un, nl, nl_prev, dt)

Build RHS for CNAB2 IMEX: rhs = un/dt + (3/2)·nl − (1/2)·nl_prev
(Linear explicit contribution omitted to match existing operator splitting.)
"""
function build_rhs_cnab2!(rhs::SHTnsSpectralField{T}, un::SHTnsSpectralField{T},
                          nl::SHTnsSpectralField{T}, nl_prev::SHTnsSpectralField{T}, dt::Float64) where T
    r_real = parent(rhs.data_real); r_imag = parent(rhs.data_imag)
    u_real = parent(un.data_real);  u_imag = parent(un.data_imag)
    n_real = parent(nl.data_real);  n_imag = parent(nl.data_imag)
    p_real = parent(nl_prev.data_real); p_imag = parent(nl_prev.data_imag)
    lm_range = get_local_range(un.pencil, 1)
    r_range  = get_local_range(un.pencil, 3)
    @inbounds for lm_idx in lm_range
        if lm_idx <= un.nlm
            ll = lm_idx - first(lm_range) + 1
            for r in r_range
                lr = r - first(r_range) + 1
                r_real[ll,1,lr] = u_real[ll,1,lr]/dt + (3/2)*n_real[ll,1,lr] - (1/2)*p_real[ll,1,lr]
                r_imag[ll,1,lr] = u_imag[ll,1,lr]/dt + (3/2)*n_imag[ll,1,lr] - (1/2)*p_imag[ll,1,lr]
            end
        end
    end
    return rhs
end

function solve_implicit_step!(solution::SHTnsSpectralField{T}, 
                                rhs::SHTnsSpectralField{T},
                                matrices::SHTnsImplicitMatrices{T}) where T

    # Get local data
    sol_real = parent(solution.data_real)
    sol_imag = parent(solution.data_imag)
    rhs_real = parent(rhs.data_real)
    rhs_imag = parent(rhs.data_imag)
    
    # Get local ranges
    lm_range = get_local_range(solution.pencil, 1)
    
    # Reusable buffers for RHS/solution to avoid allocations
    nr = size(rhs_real, 3)
    tmp_r = Vector{Float64}(undef, nr)
    tmp_i = Vector{Float64}(undef, nr)

    for lm_idx in lm_range
        if lm_idx <= solution.nlm
            l = solution.config.l_values[lm_idx]
            local_lm = lm_idx - first(lm_range) + 1
            
            matrix_idx = findfirst(==(l), matrices.l_values)
            if matrix_idx !== nothing
                factorization = matrices.factorizations[matrix_idx]
                
                # Solve for real part
                if local_lm <= size(rhs_real, 1)
                    @inbounds for k in 1:nr
                        tmp_r[k] = rhs_real[local_lm, 1, k]
                        tmp_i[k] = rhs_imag[local_lm, 1, k]
                    end
                    if factorization isa BandedLU
                        solve_banded!(tmp_r, factorization, tmp_r)
                        solve_banded!(tmp_i, factorization, tmp_i)
                    else
                        # Fallback to dense LU if present
                        tmp_r .= factorization \ tmp_r
                        tmp_i .= factorization \ tmp_i
                    end
                    @inbounds for k in 1:nr
                        sol_real[local_lm, 1, k] = tmp_r[k]
                        sol_imag[local_lm, 1, k] = tmp_i[k]
                    end
                end
            end
        end
    end
end


function compute_timestep_error(new_field::SHTnsSpectralField{T}, 
                               old_field::SHTnsSpectralField{T}) where T
    error = zero(Float64)
    
    # Get local data
    new_real = parent(new_field.data_real)
    new_imag = parent(new_field.data_imag)
    old_real = parent(old_field.data_real)
    old_imag = parent(old_field.data_imag)
    
    # Compute local error
    for idx in eachindex(new_real)
        diff_real = new_real[idx] - old_real[idx]
        diff_imag = new_imag[idx] - old_imag[idx]
        error += diff_real^2 + diff_imag^2
    end
    
    # Global reduction
    global_error = MPI.Allreduce(error, MPI.SUM, get_comm())
    return sqrt(global_error)
end


# Exports are handled by main module
