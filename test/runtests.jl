using Test
using Geodynamo

@testset "Geodynamo.jl" begin
    include("shtnskit_roundtrip.jl")
    include("ball_roundtrip.jl")
end
