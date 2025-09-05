#!/usr/bin/env julia

using Geodynamo
using Test

function test_fixed_and_no_flux()
    params = GeodynamoParameters(geometry=:shell, i_N=32, i_L=16, i_M=16, i_Th=32, i_Ph=64)
    set_parameters!(params)
    cfg = Geodynamo.create_shtnskit_config(lmax=i_L, mmax=i_M, nlat=i_Th, nlon=i_Ph)
    dom = Geodynamo.create_radial_domain(i_N)
    comp = Geodynamo.create_shtns_composition_field(Float64, cfg, dom)

    # Set BCs: inner fixed=1.0, outer no-flux
    Geodynamo.set_composition_boundary_conditions!(comp, :fixed, :no_flux; value_inner=1.0, value_outer=0.0)
    Geodynamo.apply_composition_boundary_conditions_spectral!(comp, dom)

    spec_r = parent(comp.spectral.data_real)
    r_range = Geodynamo.get_local_range(comp.spectral.pencil, 3)
    if 1 in r_range
        lr = 1 - first(r_range) + 1
        @test spec_r[1,1,lr] ≈ 1.0 atol=1e-12
    end
    if dom.N in r_range
        # Check derivative ~ 0 at outer boundary
        dr = Geodynamo.create_derivative_matrix(1, dom)
        prof = zeros(Float64, dom.N)
        for r in 1:dom.N
            if r in r_range
                lr2 = r - first(r_range) + 1
                if lr2 <= size(spec_r, 3)
                    prof[r] = spec_r[1,1,lr2]
                end
            end
        end
        d = zeros(Float64, dom.N)
        Geodynamo.apply_derivative_matrix!(d, dr, prof)
        @test abs(d[end]) ≤ 1e-8
    end
end

test_fixed_and_no_flux()
println("Composition BC tests completed.")

