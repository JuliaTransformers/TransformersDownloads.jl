using Test
using Pickle
using SafeTensors
using TransformersDownloads

@testset "Hugging Face Hub (network)" begin
    @testset "weights (safetensors)" begin
        weights = load_weights("mkshing/tiny-random-bert")
        @test weights isa AbstractDict
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
