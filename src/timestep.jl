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
    exp_action_krylov(Aop!, v, dt; m=20, tol=1e-8) -> y ≈ exp(dt A) v

Simple Arnoldi-based approximation of the exponential action.
"""
function exp_action_krylov(Aop!, v::Vector{T}, dt::Float64; m::Int=20, tol::Float64=1e-8) where T
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
            # Adaptive residual-based stopping criterion
            # Estimate residual: r ≈ |h_{j+1,j}| * |e_j^T exp(dt H_j) (beta*e1)|
            # Stop if r <= tol * ||exp(dt H_j) (beta*e1)||
            Hred_j = dt .* @view H[1:j, 1:j]
            e1 = zeros(T, j); e1[1] = one(T)
            y_small_j = exp(Hred_j) * (beta .* e1)
            res_est = abs(H[j+1, j]) * abs(y_small_j[end])
            if res_est <= tol * norm(y_small_j)
                kmax = j
                break
            end
        end
    end
    Hred = dt .* H[1:kmax, 1:kmax]
    e1 = zeros(T, kmax); e1[1] = one(T)
    y_small = exp(Hred) * (beta .* e1)
    return V[:, 1:kmax] * y_small
end

"""
    phi1_action_krylov(BA, LU_A, v, dt; m=20, tol=1e-8) -> y ≈ φ1(dt A) v

Compute φ1(dt A) v = A^{-1}[(exp(dt A) − I) v]/dt using Krylov exp(action) and banded solve.
"""
function phi1_action_krylov(Aop!, A_lu::BandedLU{T}, v::Vector{T}, dt::Float64; m::Int=20, tol::Float64=1e-8) where T
    ev = exp_action_krylov(Aop!, v, dt; m, tol)
    c = ev .- v
    x = copy(c)
    solve_banded!(x, A_lu, c)
    @. x = x / dt
    return x
end

"""
    eab2_update_krylov!(u, nl, nl_prev, domain, diffusivity, config, dt; m=20, tol=1e-8)

EAB2 update using Krylov exp/φ1 actions and banded LU for φ1.
"""
function eab2_update_krylov!(u::SHTnsSpectralField{T}, nl::SHTnsSpectralField{T},
                             nl_prev::SHTnsSpectralField{T}, domain::RadialDomain,
                             diffusivity::Float64, config::SHTnsKitConfig,
                             dt::Float64; m::Int=20, tol::Float64=1e-8) where T
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
            function Aop!(out, v)
                apply_banded_full!(out, A_banded, v)
                return nothing
            end
            # Real
            ur_new = exp_action_krylov(Aop!, ur, dt; m, tol)
            add_r = phi1_action_krylov(Aop!, A_lu, nrn, dt; m, tol)
            @. ur_new = ur_new + dt * add_r
            # Imag
            ui_new = exp_action_krylov(Aop!, ui, dt; m, tol)
            add_i = phi1_action_krylov(Aop!, A_lu, nin, dt; m, tol)
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
    get_eab2_alu_cache!(caches, key, ν, T, domain) -> Dict{Int,Tuple{BandedMatrix{T},BandedLU{T}}}

Retrieve or initialize a cache mapping l -> (A_banded, LU(A_banded)) for EAB2.
Reinitializes if ν or nr changed.
"""
function get_eab2_alu_cache!(caches::Dict{Symbol,Any}, key::Symbol, ν::Float64, ::Type{T}, domain::RadialDomain) where T
    entry = get(caches, key, nothing)
    nr = domain.N
    if entry === nothing || entry[:ν] != ν || entry[:nr] != nr
        entry = Dict{Symbol,Any}(:ν => ν, :nr => nr, :map => Dict{Int, Tuple{BandedMatrix{T}, BandedLU{T}}}())
        caches[key] = entry
    end
    return entry[:map]
end

"""
    eab2_update_krylov_cached!(u, nl, nl_prev, alu_map, domain, ν, config, dt; m=20, tol=1e-8)

Same as eab2_update_krylov!, but reuses cached banded A and LU per l.
"""
function eab2_update_krylov_cached!(u::SHTnsSpectralField{T}, nl::SHTnsSpectralField{T},
                                    nl_prev::SHTnsSpectralField{T}, alu_map::Dict{Int, Tuple{BandedMatrix{T}, BandedLU{T}}},
                                    domain::RadialDomain, diffusivity::Float64, config::SHTnsKitConfig,
                                    dt::Float64; m::Int=20, tol::Float64=1e-8) where T
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
            # get or build A and LU for this l
            tup = get(alu_map, l, nothing)
            if tup === nothing
                A_banded = build_banded_A(T, domain, diffusivity, l)
                A_lu = factorize_banded(A_banded)
                tup = (A_banded, A_lu)
                alu_map[l] = tup
            end
            A_banded, A_lu = tup
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
            function Aop!(out, v)
                apply_banded_full!(out, A_banded, v)
                return nothing
            end
            ur_new = exp_action_krylov(Aop!, ur, dt; m, tol)
            add_r = phi1_action_krylov(Aop!, A_lu, nrn, dt; m, tol)
            @. ur_new = ur_new + dt * add_r
            ui_new = exp_action_krylov(Aop!, ui, dt; m, tol)
            add_i = phi1_action_krylov(Aop!, A_lu, nin, dt; m, tol)
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
    
    # Compute local error with bounds checking for PencilArrays
    @inbounds for idx in eachindex(new_real, old_real)
        diff_real = new_real[idx] - old_real[idx]
        diff_imag = new_imag[idx] - old_imag[idx]
        error += diff_real^2 + diff_imag^2
    end
    
    # Global reduction across all MPI processes
    global_error = MPI.Allreduce(error, MPI.SUM, get_comm())
    return sqrt(global_error)
end

"""
    synchronize_pencil_transforms!(field::SHTnsSpectralField{T}) where T

Ensure all pending PencilFFTs operations are completed and data is consistent across processes.
"""
function synchronize_pencil_transforms!(field::SHTnsSpectralField{T}) where T
    # Synchronize data across pencil decomposition
    MPI.Barrier(get_comm())
    return field
end

"""
    validate_mpi_consistency!(field::SHTnsSpectralField{T}) where T

Check that spectral field data is consistent across MPI processes after time stepping.
"""
function validate_mpi_consistency!(field::SHTnsSpectralField{T}) where T
    comm = get_comm()
    rank = get_rank()
    nprocs = get_nprocs()
    
    if nprocs > 1
        # Check a few sample values for consistency
        real_data = parent(field.data_real)
        imag_data = parent(field.data_imag)
        
        # Sample first few elements
        n_samples = min(5, length(real_data))
        local_samples_real = Vector{T}(undef, n_samples)
        local_samples_imag = Vector{T}(undef, n_samples)
        
        @inbounds for i in 1:n_samples
            local_samples_real[i] = real_data[i]
            local_samples_imag[i] = imag_data[i]
        end
        
        # Gather samples from all processes
        all_samples_real = MPI.Allgather(local_samples_real, comm)
        all_samples_imag = MPI.Allgather(local_samples_imag, comm)
        
        # Check consistency on rank 0
        if rank == 0
            max_diff_real = zero(T)
            max_diff_imag = zero(T)
            
            for proc in 2:nprocs
                for i in 1:n_samples
                    diff_real = abs(all_samples_real[(proc-1)*n_samples + i] - all_samples_real[i])
                    diff_imag = abs(all_samples_imag[(proc-1)*n_samples + i] - all_samples_imag[i])
                    max_diff_real = max(max_diff_real, diff_real)
                    max_diff_imag = max(max_diff_imag, diff_imag)
                end
            end
            
            # Warn if inconsistency detected
            if max_diff_real > 1e-12 || max_diff_imag > 1e-12
                @warn "MPI data inconsistency detected: max_diff_real=$max_diff_real, max_diff_imag=$max_diff_imag"
            end
        end
    end
    
    return field
end


# ============================================================================
# Exponential 2nd Order Runge-Kutta (ERK2) Implementation
# ============================================================================

"""
    ERK2Cache{T}

Cached data structure for Exponential 2nd Order Runge-Kutta method.
Stores precomputed matrix exponentials and φ functions for each spherical harmonic mode.
"""
struct ERK2Cache{T}
    dt::Float64
    l_values::Vector{Int}
    
    # Matrix exponentials: exp(dt/2 * A_l) and exp(dt * A_l)
    E_half::Vector{Matrix{T}}     # exp(dt/2 * A_l) per l
    E_full::Vector{Matrix{T}}     # exp(dt * A_l) per l
    
    # φ functions for ERK2
    phi1_half::Vector{Matrix{T}}  # φ1(dt/2 * A_l) per l
    phi1_full::Vector{Matrix{T}}  # φ1(dt * A_l) per l  
    phi2_full::Vector{Matrix{T}}  # φ2(dt * A_l) per l
    
    # Krylov method parameters
    use_krylov::Bool
    krylov_m::Int
    krylov_tol::Float64
    
    # MPI-aware caching for distributed operations
    mpi_consistent::Bool
end

"""
    create_erk2_cache(config, domain, diffusivity, dt; use_krylov=false, m=20, tol=1e-8)

Create ERK2 cache with precomputed matrix functions for all spherical harmonic modes.
"""
function create_erk2_cache(::Type{T}, config::SHTnsKitConfig, domain::RadialDomain,
                          diffusivity::Float64, dt::Float64;
                          use_krylov::Bool=false, m::Int=20, tol::Float64=1e-8) where T
    
    lap = create_radial_laplacian(domain)
    nr = domain.N
    r_inv2 = @views domain.r[1:nr, 2]
    lvals = unique(config.l_values)
    
    E_half = Matrix{T}[]
    E_full = Matrix{T}[]
    phi1_half = Matrix{T}[]
    phi1_full = Matrix{T}[]
    phi2_full = Matrix{T}[]
    
    if get_rank() == 0
        @info "Creating ERK2 cache for $(length(lvals)) l-modes with $(use_krylov ? "Krylov" : "dense") methods"
    end
    
    for l in lvals
        # Build A_l = diffusivity * (d²/dr² + (2/r)d/dr - l(l+1)/r²)
        Adata = diffusivity .* lap.data
        Adense = banded_to_dense(BandedMatrix(Adata, i_KL, nr))
        lfac = Float64(l * (l + 1))
        
        @inbounds for n in 1:nr
            Adense[n, n] -= diffusivity * lfac * r_inv2[n]
        end
        
        if use_krylov
            # For large problems, we'll use Krylov methods during timestepping
            # Store only the operator for action-based computation
            push!(E_half, Adense)  # Store A for later Krylov action
            push!(E_full, Adense)
            push!(phi1_half, Adense)
            push!(phi1_full, Adense)
            push!(phi2_full, Adense)
        else
            # Dense computation of matrix functions
            Adt_half = (dt/2) .* Adense
            Adt_full = dt .* Adense
            
            # Compute exp(dt/2 * A) and exp(dt * A)
            E_half_l = exp(Adt_half)
            E_full_l = exp(Adt_full)
            push!(E_half, Matrix{T}(E_half_l))
            push!(E_full, Matrix{T}(E_full_l))
            
            # Compute φ1 functions: φ1(z) = (exp(z) - I) / z
            phi1_half_l = compute_phi1_function(Adt_half, E_half_l)
            phi1_full_l = compute_phi1_function(Adt_full, E_full_l)
            push!(phi1_half, Matrix{T}(phi1_half_l))
            push!(phi1_full, Matrix{T}(phi1_full_l))
            
            # Compute φ2 function: φ2(z) = (exp(z) - I - z) / z²
            phi2_full_l = compute_phi2_function(Adt_full, E_full_l)
            push!(phi2_full, Matrix{T}(phi2_full_l))
        end
    end
    
    # Ensure MPI consistency
    MPI.Barrier(get_comm())
    
    return ERK2Cache{T}(dt, lvals, E_half, E_full, phi1_half, phi1_full, phi2_full,
                       use_krylov, m, tol, true)
end

"""
    compute_phi1_function(A, expA)

Compute φ1(A) = A^(-1) * (exp(A) - I) efficiently.
"""
function compute_phi1_function(A::Matrix{T}, expA::Matrix{T}) where T
    nr = size(A, 1)
    I_mat = Matrix{T}(I, nr, nr)
    
    # φ1(A) = A^(-1) * (exp(A) - I)
    diff = expA - I_mat
    
    # Use lu factorization for stable inversion
    try
        lu_A = lu(A)
        return lu_A \ diff
    catch
        # Fallback for singular/near-singular matrices
        return pinv(A) * diff
    end
end

"""
    compute_phi2_function(A, expA)

Compute φ2(A) = A^(-2) * (exp(A) - I - A) efficiently.
"""
function compute_phi2_function(A::Matrix{T}, expA::Matrix{T}) where T
    nr = size(A, 1)
    I_mat = Matrix{T}(I, nr, nr)
    
    # φ2(A) = A^(-2) * (exp(A) - I - A)
    diff = expA - I_mat - A
    
    try
        lu_A = lu(A)
        temp = lu_A \ diff
        return lu_A \ temp  # A^(-1) * A^(-1) * diff
    catch
        # Fallback
        A_inv = pinv(A)
        return A_inv * A_inv * diff
    end
end

"""
    erk2_step!(u, nl_current, nl_prev, cache, config, dt)

Perform one ERK2 timestep: u^{n+1} = exp(dt*A)*u^n + dt*φ1(dt*A)*nl^n + dt²*φ2(dt*A)*(nl^n - nl^{n-1})/dt

This implements the exponential 2nd order Runge-Kutta method with MPI and PencilArrays support.
"""
function erk2_step!(u::SHTnsSpectralField{T}, nl_current::SHTnsSpectralField{T},
                   nl_prev::SHTnsSpectralField{T}, cache::ERK2Cache{T},
                   config::SHTnsKitConfig, dt::Float64) where T
    
    u_real = parent(u.data_real)
    u_imag = parent(u.data_imag)
    nl_real = parent(nl_current.data_real)
    nl_imag = parent(nl_current.data_imag)
    np_real = parent(nl_prev.data_real)
    np_imag = parent(nl_prev.data_imag)
    
    lm_range = get_local_range(u.pencil, 1)
    r_range = get_local_range(u.pencil, 3)
    nr = size(u_real, 3)
    
    comm = get_comm()
    multi = MPI.Comm_size(comm) > 1
    
    # Process each spherical harmonic mode
    for lm_idx in lm_range
        if lm_idx <= u.nlm
            l = config.l_values[lm_idx]
            ll = lm_idx - first(lm_range) + 1
            
            # Find cache index for this l
            cache_idx = findfirst(==(l), cache.l_values)
            if cache_idx === nothing
                continue
            end
            
            # Assemble full radial profiles
            ur = zeros(T, nr); ui = zeros(T, nr)
            nr_cur = zeros(T, nr); ni_cur = zeros(T, nr)
            nr_prev = zeros(T, nr); ni_prev = zeros(T, nr)
            
            @inbounds for r in r_range
                lr = r - first(r_range) + 1
                if lr <= size(u_real, 3)
                    ur[r] = u_real[ll, 1, lr]
                    ui[r] = u_imag[ll, 1, lr]
                    nr_cur[r] = nl_real[ll, 1, lr]
                    ni_cur[r] = nl_imag[ll, 1, lr]
                    nr_prev[r] = np_real[ll, 1, lr]
                    ni_prev[r] = np_imag[ll, 1, lr]
                end
            end
            
            # MPI synchronization for distributed data
            if multi
                MPI.Allreduce!(ur, MPI.SUM, comm)
                MPI.Allreduce!(ui, MPI.SUM, comm)
                MPI.Allreduce!(nr_cur, MPI.SUM, comm)
                MPI.Allreduce!(ni_cur, MPI.SUM, comm)
                MPI.Allreduce!(nr_prev, MPI.SUM, comm)
                MPI.Allreduce!(ni_prev, MPI.SUM, comm)
            end
            
            # Apply ERK2 formula
            if cache.use_krylov
                # Use Krylov-based ERK2 for large problems
                ur_new, ui_new = erk2_krylov_step(ur, ui, nr_cur, ni_cur, nr_prev, ni_prev,
                                                 cache.E_full[cache_idx], dt, cache.krylov_m, cache.krylov_tol)
            else
                # Use precomputed matrices for ERK2
                ur_new, ui_new = erk2_matrix_step(ur, ui, nr_cur, ni_cur, nr_prev, ni_prev,
                                                 cache.E_full[cache_idx], cache.phi1_full[cache_idx], 
                                                 cache.phi2_full[cache_idx], dt)
            end
            
            # Scatter back to local data
            @inbounds for r in r_range
                lr = r - first(r_range) + 1
                if lr <= size(u_real, 3)
                    u_real[ll, 1, lr] = ur_new[r]
                    u_imag[ll, 1, lr] = ui_new[r]
                end
            end
        end
    end
    
    # Synchronize across PencilArrays
    synchronize_pencil_transforms!(u)
    
    return u
end

"""
    erk2_matrix_step(u_r, u_i, nl_r, nl_i, np_r, np_i, E, phi1, phi2, dt)

ERK2 step using precomputed matrix exponentials.
"""
function erk2_matrix_step(u_r::Vector{T}, u_i::Vector{T}, 
                         nl_r::Vector{T}, nl_i::Vector{T},
                         np_r::Vector{T}, np_i::Vector{T},
                         E::Matrix{T}, phi1::Matrix{T}, phi2::Matrix{T}, dt::Float64) where T
    
    # ERK2 formula: u^{n+1} = exp(dt*A)*u^n + dt*φ1(dt*A)*nl^n + dt²*φ2(dt*A)*(nl^n - nl^{n-1})/dt
    
    # Linear part: exp(dt*A) * u^n
    ur_new = E * u_r
    ui_new = E * u_i
    
    # First nonlinear part: dt * φ1(dt*A) * nl^n
    @. ur_new += dt * (phi1 * nl_r)
    @. ui_new += dt * (phi1 * nl_i)
    
    # Second nonlinear part: dt² * φ2(dt*A) * (nl^n - nl^{n-1})/dt
    diff_r = nl_r .- np_r
    diff_i = nl_i .- np_i
    @. ur_new += dt * (phi2 * diff_r)  # dt² * φ2 * diff / dt = dt * φ2 * diff
    @. ui_new += dt * (phi2 * diff_i)
    
    return ur_new, ui_new
end

"""
    erk2_krylov_step(u_r, u_i, nl_r, nl_i, np_r, np_i, A, dt, m, tol)

ERK2 step using Krylov methods for matrix function actions.
"""
function erk2_krylov_step(u_r::Vector{T}, u_i::Vector{T},
                         nl_r::Vector{T}, nl_i::Vector{T}, 
                         np_r::Vector{T}, np_i::Vector{T},
                         A::Matrix{T}, dt::Float64, m::Int, tol::Float64) where T
    
    nr = length(u_r)
    
    # Define operator action for A
    function Aop!(out, v)
        mul!(out, A, v)
        return nothing
    end
    
    # Compute exp(dt*A) * u using Krylov method
    ur_new = exp_action_krylov(Aop!, u_r, dt; m, tol)
    ui_new = exp_action_krylov(Aop!, u_i, dt; m, tol)
    
    # Compute φ1(dt*A) * nl using Krylov method
    phi1_nl_r = phi1_action_krylov(Aop!, A, nl_r, dt; m, tol)
    phi1_nl_i = phi1_action_krylov(Aop!, A, nl_i, dt; m, tol)
    
    # Add first nonlinear contribution
    @. ur_new += dt * phi1_nl_r
    @. ui_new += dt * phi1_nl_i
    
    # Compute φ2(dt*A) * (nl - np) using Krylov method  
    diff_r = nl_r .- np_r
    diff_i = nl_i .- np_i
    phi2_diff_r = phi2_action_krylov(Aop!, A, diff_r, dt; m, tol)
    phi2_diff_i = phi2_action_krylov(Aop!, A, diff_i, dt; m, tol)
    
    # Add second nonlinear contribution
    @. ur_new += dt * phi2_diff_r
    @. ui_new += dt * phi2_diff_i
    
    return ur_new, ui_new
end

"""
    phi1_action_krylov(Aop!, A, v, dt; m=20, tol=1e-8)

Compute φ1(dt*A) * v using Krylov methods where φ1(z) = (exp(z) - I) / z.
"""
function phi1_action_krylov(Aop!, A::Matrix{T}, v::Vector{T}, dt::Float64; 
                           m::Int=20, tol::Float64=1e-8) where T
    
    # φ1(dt*A) * v = A^{-1} * (exp(dt*A) - I) * v
    exp_v = exp_action_krylov(Aop!, v, dt; m, tol)
    diff_v = exp_v .- v
    
    # Solve A * result = diff_v for result
    result = A \ diff_v
    return result
end

"""
    phi2_action_krylov(Aop!, A, v, dt; m=20, tol=1e-8)

Compute φ2(dt*A) * v using Krylov methods where φ2(z) = (exp(z) - I - z) / z².
"""
function phi2_action_krylov(Aop!, A::Matrix{T}, v::Vector{T}, dt::Float64;
                           m::Int=20, tol::Float64=1e-8) where T
    
    # φ2(dt*A) * v = A^{-2} * (exp(dt*A) - I - dt*A) * v
    exp_v = exp_action_krylov(Aop!, v, dt; m, tol)
    A_v = A * v
    diff_v = exp_v .- v .- (dt .* A_v)
    
    # Solve A * (A * result) = diff_v for result
    temp = A \ diff_v
    result = A \ temp
    return result
end

"""
    get_erk2_cache!(caches, key, diffusivity, config, domain, dt; use_krylov=false)

Retrieve or create ERK2 cache with automatic invalidation when parameters change.
"""
function get_erk2_cache!(caches::Dict{Symbol,Any}, key::Symbol, diffusivity::Float64,
                        ::Type{T}, config::SHTnsKitConfig, domain::RadialDomain, dt::Float64;
                        use_krylov::Bool=false, m::Int=20, tol::Float64=1e-8) where T
    
    entry = get(caches, key, nothing)
    nr = domain.N
    
    # Check if cache needs to be rebuilt
    if entry === nothing || 
       get(entry, :diffusivity, nothing) != diffusivity ||
       get(entry, :nr, nothing) != nr ||
       get(entry, :dt, nothing) != dt ||
       !get(entry, :mpi_consistent, false)
        
        if get_rank() == 0
            @info "Creating new ERK2 cache for $key (ν=$diffusivity, nr=$nr, dt=$dt)"
        end
        
        cache = create_erk2_cache(T, config, domain, diffusivity, dt; 
                                use_krylov, m, tol)
        
        entry = Dict{Symbol,Any}(
            :cache => cache,
            :diffusivity => diffusivity,
            :nr => nr,
            :dt => dt,
            :mpi_consistent => true
        )
        caches[key] = entry
    end
    
    return entry[:cache]
end

# Exports are handled by main module
