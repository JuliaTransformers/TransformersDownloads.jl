using Test
using Pickle
using SafeTensors
using TransformersDownloads

if !isdefined(@__MODULE__, :TD)
    const TD = TransformersDownloads
end

@testset "Hugging Face Hub (network)" begin
    @testset "weights (single safetensors)" begin
        weights = load_weights("hf-internal-testing/tiny-random-bert")
        @test weights isa SafeTensors.SafeTensor
        @test weights isa AbstractDict{String,<:AbstractArray}
        @test !isempty(weights)
        @test all(v -> v isa AbstractArray, values(weights))
    end

    @testset "weights (sharded safetensors)" begin
        weights = load_weights("hf-internal-testing/tiny-random-bert-sharded")
        @test weights isa TD.ShardedStateDict
        @test weights isa AbstractDict{String,<:AbstractArray}
        @test !isempty(weights)
        @test all(v -> v isa AbstractArray, values(weights))
    end

    @testset "weights (pickle extension)" begin
        weights = load_weights("hf-internal-testing/tiny-random-DistilBertModel")
        @test weights isa AbstractDict
        @test !isempty(weights)
        @test any(occursin("embeddings", k) for k in keys(weights))
    end

    @testset "config" begin
        config = load_config("hf-internal-testing/tiny-random-DistilBertModel")
        @test config isa AbstractDict
        @test haskey(config, "vocab_size")
        @test haskey(config, "n_layers")
    end

    @testset "vocab and tokenizer" begin
        vocab = load_vocab("google-bert/bert-base-uncased")
        @test vocab isa Vector{String}
        @test "[CLS]" in vocab

        tokenizer = load_tokenizer("google-bert/bert-base-uncased")
        @test tokenizer isa AbstractDict
        @test haskey(tokenizer, "model")
    end
end
