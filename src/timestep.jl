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
    build_banded_A(T, domain, diffusivity, l) -> BandedMatrix{T}

Construct banded A = ν*(d²/dr² + (2/r)d/dr − l(l+1)/r²) in banded storage.
"""
function build_banded_A(::Type{T}, domain::RadialDomain, diffusivity::Float64, l::Int) where T
    lap = create_radial_laplacian(domain)
    data = diffusivity .* lap.data
    nr = domain.N
    r_inv2 = @views domain.r[1:nr, 2]
    lfac = Float64(l * (l + 1))
    @inbounds for n in 1:nr
        data[i_KL + 1, n] -= diffusivity * lfac * r_inv2[n]
    end
    return BandedMatrix{T}(Matrix{T}(data), i_KL, nr)
end

"""
    apply_banded_full!(out, B, v)

Apply banded matrix to full vector.
"""
function apply_banded_full!(out::Vector{T}, B::BandedMatrix{T}, v::Vector{T}) where T
    fill!(out, zero(T))
    N = B.size; bw = B.bandwidth
    @inbounds for j in 1:N
        for i in max(1, j - bw):min(N, j + bw)
            row = bw + 1 + i - j
            if 1 <= row <= 2*bw+1
                out[i] += B.data[row, j] * v[j]
            end
        end
    end
    return out
end

"""
    exp_action_krylov(Aop!, v, dt; m=20) -> y ≈ exp(dt A) v

Simple Arnoldi-based approximation of the exponential action.
"""
function exp_action_krylov(Aop!, v::Vector{T}, dt::Float64; m::Int=20) where T
    n = length(v)
    V = Matrix{T}(undef, n, m)
    H = zeros(T, m, m)
    beta = norm(v)
    if beta == 0
        return zeros(T, n)
    end
    V[:, 1] = v / beta
    w = similar(v)
    kmax = m
    for j in 1:m
        Aop!(w, view(V, :, j))
        for i in 1:j
            H[i, j] = dot(view(V, :, i), w)
            @. w = w - H[i, j] * V[:, i]
        end
        if j < m
            H[j+1, j] = norm(w)
            if H[j+1, j] == 0
                kmax = j
                break
            end
            V[:, j+1] = w / H[j+1, j]
        end
    end
    Hred = dt .* H[1:kmax, 1:kmax]
    e1 = zeros(T, kmax); e1[1] = one(T)
    y_small = exp(Hred) * (beta .* e1)
    return V[:, 1:kmax] * y_small
end

"""
    phi1_action_krylov(BA, LU_A, v, dt; m=20) -> y ≈ φ1(dt A) v

Compute φ1(dt A) v = A^{-1}[(exp(dt A) − I) v]/dt using Krylov exp(action) and banded solve.
"""
function phi1_action_krylov(Aop!, A_lu::BandedLU{T}, v::Vector{T}, dt::Float64; m::Int=20) where T
    ev = exp_action_krylov(Aop!, v, dt; m)
    c = ev .- v
    x = copy(c)
    solve_banded!(x, A_lu, c)
    @. x = x / dt
    return x
end

"""
    eab2_update_krylov!(u, nl, nl_prev, domain, diffusivity, config, dt; m=20)

EAB2 update using Krylov exp/φ1 actions and banded LU for φ1.
"""
function eab2_update_krylov!(u::SHTnsSpectralField{T}, nl::SHTnsSpectralField{T},
                             nl_prev::SHTnsSpectralField{T}, domain::RadialDomain,
                             diffusivity::Float64, config::SHTnsKitConfig,
                             dt::Float64; m::Int=20) where T
    u_real = parent(u.data_real); u_imag = parent(u.data_imag)
    n_real = parent(nl.data_real); n_imag = parent(nl.data_imag)
    p_real = parent(nl_prev.data_real); p_imag = parent(nl_prev.data_imag)
    lm_range = get_local_range(u.pencil, 1)
    r_range  = get_local_range(u.pencil, 3)
    nr = domain.N
    comm = get_comm()
    multi = MPI.Comm_size(comm) > 1
    for lm_idx in lm_range
        if lm_idx <= u.nlm
            l = config.l_values[lm_idx]
            ll = lm_idx - first(lm_range) + 1
            A_banded = build_banded_A(T, domain, diffusivity, l)
            A_lu = factorize_banded(A_banded)
            # Assembled full vectors
            ur = zeros(T, nr); ui = zeros(T, nr)
            nrn = zeros(T, nr); nin = zeros(T, nr)
            @inbounds for r in r_range
                lr = r - first(r_range) + 1
                if lr <= size(u_real, 3)
                    ur[r] = u_real[ll,1,lr]; ui[r] = u_imag[ll,1,lr]
                    nrn[r] = (3/2)*n_real[ll,1,lr] - (1/2)*p_real[ll,1,lr]
                    nin[r] = (3/2)*n_imag[ll,1,lr] - (1/2)*p_imag[ll,1,lr]
                end
            end
            if multi
                MPI.Allreduce!(ur, MPI.SUM, comm)
                MPI.Allreduce!(ui, MPI.SUM, comm)
                MPI.Allreduce!(nrn, MPI.SUM, comm)
                MPI.Allreduce!(nin, MPI.SUM, comm)
            end
            # Define Aop! using banded apply
            tmp = zeros(T, nr)
            Aop!(out, v) = (apply_banded_full!(out, A_banded, v); nothing)
            # Real
            ur_new = exp_action_krylov(x->Aop!(tmp, x), ur, dt; m)
            add_r = phi1_action_krylov(x->Aop!(tmp, x), A_lu, nrn, dt; m)
            @. ur_new = ur_new + dt * add_r
            # Imag
            ui_new = exp_action_krylov(x->Aop!(tmp, x), ui, dt; m)
            add_i = phi1_action_krylov(x->Aop!(tmp, x), A_lu, nin, dt; m)
            @. ui_new = ui_new + dt * add_i
            # Scatter back
            @inbounds for r in r_range
                lr = r - first(r_range) + 1
                if lr <= size(u_real, 3)
                    u_real[ll,1,lr] = ur_new[r]
                    u_imag[ll,1,lr] = ui_new[r]
                end
            end
        end
    end
    return u
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
    nr_full = size(etd.E[1], 1)
    comm = get_comm()
    multi = MPI.Comm_size(comm) > 1
    # Build map from lm_idx to l index in etd
    for lm_idx in lm_range
        if lm_idx <= u.nlm
            l = config.l_values[lm_idx]
            lpos = findfirst(==(l), etd.l_values)
            E = etd.E[lpos]
            P1 = etd.phi1[lpos]
            ll = lm_idx - first(lm_range) + 1
            # Assemble full radial vectors
            ur = zeros(T, nr_full); ui = zeros(T, nr_full)
            nrn = zeros(T, nr_full); nin = zeros(T, nr_full)
            @inbounds for r in r_range
                lr = r - first(r_range) + 1
                if lr <= size(u_real, 3)
                    ur[r] = u_real[ll, 1, lr]
                    ui[r] = u_imag[ll, 1, lr]
                    nrn[r] = (3/2)*n_real[ll,1,lr] - (1/2)*p_real[ll,1,lr]
                    nin[r] = (3/2)*n_imag[ll,1,lr] - (1/2)*p_imag[ll,1,lr]
                end
            end
            if multi
                MPI.Allreduce!(ur, MPI.SUM, comm)
                MPI.Allreduce!(ui, MPI.SUM, comm)
                MPI.Allreduce!(nrn, MPI.SUM, comm)
                MPI.Allreduce!(nin, MPI.SUM, comm)
            end
            ur_new = E*ur + dt*(P1*nrn)
            ui_new = E*ui + dt*(P1*nin)
            # Scatter back to local slab
            @inbounds for r in r_range
                lr = r - first(r_range) + 1
                if lr <= size(u_real, 3)
                    u_real[ll,1,lr] = ur_new[r]
                    u_imag[ll,1,lr] = ui_new[r]
                end
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
