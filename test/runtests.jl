using Test
using TransformersDownloads

@testset "TransformersDownloads.jl" begin
    include("test_local.jl")
    include("test_quality.jl")

    if get(ENV, "TD_TEST_HUB", "false") == "true"
        include("test_network.jl")
    else
        @info "Skipping network tests. Set TD_TEST_HUB=true to run them."
    end
end
