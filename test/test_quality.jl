using Test
using Aqua
using JET
using TransformersDownloads

if !isdefined(@__MODULE__, :TD)
    const TD = TransformersDownloads
end

@testset "quality" begin
    # `persistent_tasks` precompiles the whole test environment and walks every
    # manifest entry; it fails on unrelated packages whose depot copy lacks a
    # Project.toml, so it is disabled here.
    Aqua.test_all(TransformersDownloads; persistent_tasks=false)

    @testset "JET type stability" begin
        JET.@test_opt TD.vocab_format("vocab.txt")
        JET.@test_opt TD._weights_kind_from_name("model.safetensors")
        JET.@test_opt TD.is_path_like("./model")
        JET.@test_opt TD.classify_source("/")
    end
end
