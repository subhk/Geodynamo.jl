using Test
using Geodynamo

@testset "Geodynamo.jl" begin
    include("shtnskit_roundtrip.jl")
    include("ball_roundtrip.jl")
    include("ball_finiteness.jl")
    include("shell_boundaries.jl")
end
