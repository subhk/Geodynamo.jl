#!/usr/bin/env julia

using Geodynamo
using Test

function profile_for_mode(spec_arr, lm_idx, r_range, nr)
    prof = zeros(Float64, nr)
    for r in 1:nr
        if r in r_range
            lr = r - first(r_range) + 1
            if lr <= size(spec_arr, 3)
                prof[r] = spec_arr[lm_idx, 1, lr]
            end
        end
    end
    return prof
end

function test_dirichlet()
    params = GeodynamoParameters(geometry=:shell, i_N=32, i_L=16, i_M=16, i_Th=32, i_Ph=64)
    set_parameters!(params)
    cfg = Geodynamo.create_shtnskit_config(lmax=i_L, mmax=i_M, nlat=i_Th, nlon=i_Ph)
    dom = Geodynamo.create_radial_domain(i_N)
    tf = Geodynamo.create_shtns_temperature_field(Float64, cfg, dom)

    # Set Dirichlet at both boundaries
    Geodynamo.set_boundary_conditions!(tf; inner_bc_type=1, outer_bc_type=1, inner_value=1.0, outer_value=0.0)

    # Apply BCs in spectral space
    Geodynamo.apply_temperature_boundary_conditions_spectral!(tf, dom)

    # Check spectral boundary planes set for l=0,m=0
    l0m0 = 1  # by construction order in this config
    r_range = Geodynamo.get_local_range(tf.spectral.pencil, 3)
    if 1 in r_range
        lr = 1 - first(r_range) + 1
        @test parent(tf.spectral.data_real)[l0m0,1,lr] ≈ 1.0 atol=1e-12
    end
    if dom.N in r_range
        lr = dom.N - first(r_range) + 1
        @test parent(tf.spectral.data_real)[l0m0,1,lr] ≈ 0.0 atol=1e-12
    end
end

function test_flux()
    params = GeodynamoParameters(geometry=:shell, i_N=32, i_L=16, i_M=16, i_Th=32, i_Ph=64)
    set_parameters!(params)
    cfg = Geodynamo.create_shtnskit_config(lmax=i_L, mmax=i_M, nlat=i_Th, nlon=i_Ph)
    dom = Geodynamo.create_radial_domain(i_N)
    tf = Geodynamo.create_shtns_temperature_field(Float64, cfg, dom)

    # Flux at both boundaries (uses get_flux_value: inner=+1, outer=-1 for l=0,m=0)
    Geodynamo.set_boundary_conditions!(tf; inner_bc_type=2, outer_bc_type=2)

    # Apply BCs
    Geodynamo.apply_temperature_boundary_conditions_spectral!(tf, dom)

    # Reconstruct profile for l=0,m=0 and check derivatives at boundaries
    l0m0 = 1
    r_range = Geodynamo.get_local_range(tf.spectral.pencil, 3)
    spec_r = parent(tf.spectral.data_real)
    prof = profile_for_mode(spec_r, l0m0, r_range, dom.N)
    dr = Geodynamo.create_derivative_matrix(1, dom)
    d = zeros(Float64, dom.N)
    Geodynamo.apply_derivative_matrix!(d, dr, prof)
    @test d[1] ≈ 1.0 atol=1e-3
    @test d[end] ≈ -1.0 atol=1e-3
end

test_dirichlet()
test_flux()
println("Thermal BC tests completed.")

