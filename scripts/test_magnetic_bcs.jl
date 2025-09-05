#!/usr/bin/env julia

using Geodynamo
using Test

function main()
    # Shell geometry to exercise inner/outer boundaries distinctly
    params = GeodynamoParameters(geometry=:shell, i_N=32, i_Nic=16, i_L=16, i_M=16, i_Th=32, i_Ph=64)
    set_parameters!(params)

    # Build simple state components to access domains
    state = initialize_shtns_simulation(Float64; include_composition=false)

    # Randomize magnetic spectral fields (outer and inner core)
    rand!(parent(state.magnetic.toroidal.data_real))
    rand!(parent(state.magnetic.toroidal.data_imag))
    rand!(parent(state.magnetic.poloidal.data_real))
    rand!(parent(state.magnetic.poloidal.data_imag))

    rand!(parent(state.magnetic.ic_toroidal.data_real))
    rand!(parent(state.magnetic.ic_toroidal.data_imag))
    rand!(parent(state.magnetic.ic_poloidal.data_real))
    rand!(parent(state.magnetic.ic_poloidal.data_imag))

    # Apply magnetic BCs
    apply_magnetic_boundary_conditions!(state.magnetic, state.oc_domain, state.ic_domain)

    # Check outer boundary: insulating (∂B/∂r ≈ 0 using derivative operator)
    tor = parent(state.magnetic.toroidal.data_real)
    pol = parent(state.magnetic.poloidal.data_real)
    lm_range = range_local(state.magnetic.toroidal.pencil, 1)
    r_range  = range_local(state.magnetic.toroidal.pencil, 3)
    if state.oc_domain.N in r_range
        rloc = state.oc_domain.N - first(r_range) + 1
        # Derivative operator check
        dr = Geodynamo.create_derivative_matrix(1, state.oc_domain)
        for lm_idx in lm_range
            if lm_idx <= state.magnetic.toroidal.nlm
                ll = lm_idx - first(lm_range) + 1
                # Reconstruct profiles and verify derivative at outer boundary ≈ 0
                prof_t = [ (r in r_range && (r - first(r_range) + 1) <= size(tor,3)) ? tor[ll,1,r - first(r_range) + 1] : 0.0 for r in 1:state.oc_domain.N ]
                prof_p = [ (r in r_range && (r - first(r_range) + 1) <= size(pol,3)) ? pol[ll,1,r - first(r_range) + 1] : 0.0 for r in 1:state.oc_domain.N ]
                d_t = zeros(Float64, state.oc_domain.N); Geodynamo.apply_derivative_matrix!(d_t, dr, prof_t)
                d_p = zeros(Float64, state.oc_domain.N); Geodynamo.apply_derivative_matrix!(d_p, dr, prof_p)
                @test abs(d_t[end]) ≤ 1e-8
                @test abs(d_p[end]) ≤ 1e-8
            end
        end
    end

    # Check inner boundary continuity: match inner core first plane and derivative continuity (poloidal)
    if 1 in r_range
        rloc = 1 - first(r_range) + 1
        ic_tor = parent(state.magnetic.ic_toroidal.data_real)
        ic_pol = parent(state.magnetic.ic_poloidal.data_real)
        dr_oc = Geodynamo.create_derivative_matrix(1, state.oc_domain)
        dr_ic = Geodynamo.create_derivative_matrix(1, state.ic_domain)
        for lm_idx in lm_range
            if lm_idx <= state.magnetic.toroidal.nlm
                ll = lm_idx - first(lm_range) + 1
                @test tor[ll,1,rloc] ≈ ic_tor[ll,1,1] atol=1e-12
                @test pol[ll,1,rloc] ≈ ic_pol[ll,1,1] atol=1e-12
                # Derivative continuity check for poloidal
                prof_oc = [ (r in r_range && (r - first(r_range) + 1) <= size(pol,3)) ? pol[ll,1,r - first(r_range) + 1] : 0.0 for r in 1:state.oc_domain.N ]
                # Build a simple IC profile vector from the first radial plane (limited by local data)
                prof_ic = [ ic_pol[ll,1,1] for _ in 1:state.ic_domain.N ]
                d_oc = zeros(Float64, state.oc_domain.N); Geodynamo.apply_derivative_matrix!(d_oc, dr_oc, prof_oc)
                d_ic = zeros(Float64, state.ic_domain.N); Geodynamo.apply_derivative_matrix!(d_ic, dr_ic, prof_ic)
                @test abs(d_oc[1] - d_ic[1]) ≤ 1e-6
            end
        end
    end

    println("Magnetic BC tests completed.")
end

main()
