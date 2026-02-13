using Test
using TransformersDownloads
using SafeTensors
using Pickle

@testset "TransformersDownloads.jl" begin

    safetensors_model = "mkshing/tiny-random-bert" # Attempting a different tiny model
    # If not found, fall back to "bert-base-uncased" (but only config?)

    model_name_pickle = "hf-internal-testing/tiny-random-DistilBertModel"

    @testset "Config" begin
        config = load_config(model_name_pickle)
        @test config isa AbstractDict
        @test haskey(config, "vocab_size")
        @test haskey(config, "n_layers")
    end

    @testset "Weights (Pickle - Extension)" begin
        # This model has pytorch_model.bin
        weights = load_weights(model_name_pickle)
        @test weights isa AbstractDict
        @test !isempty(weights)
        # Check a key (random check, assuming distilbert keys)
        @test any(occursin("embeddings", k) for k in keys(weights))
    end

end
