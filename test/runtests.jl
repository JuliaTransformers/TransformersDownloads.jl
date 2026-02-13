using Test
using TransformersDownloads
using SafeTensors
using Pickle

@testset "TransformersDownloads.jl" begin
    # 1. Test a model with SafeTensors (bert-base-uncased)
    # We use a known model that has safetensors.
    # Note: Downloading ~400MB might be slow. Is there a smaller one?
    # hf-internal-testing/tiny-bert-for-token-classification has safetensors?
    # Let's try "hf-internal-testing/tiny-random-BertModel" again? No, it didnt have it.
    # "HuggingFaceM4/tiny-random-Llama-3-8b" (too big structure?)
    # "mkshing/tiny-random-bert-safetensors" -> Let's check this one or similar?
    # Actually, let's just use a real tiny model that I verify has safetensors.
    # "diffusers/tiny-stable-diffusion-torch" has models?
    # Let's try "google-bert/bert-base-uncased" for config, but maybe skip weights if too big?
    # OR, we verify extension loading with the tiny-random-DistilBertModel (which has Pickle).

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
        weights = load_state_dict(model_name_pickle)
        @test weights isa AbstractDict
        @test !isempty(weights)
        # Check a key (random check, assuming distilbert keys)
        @test any(occursin("embeddings", k) for k in keys(weights))
    end

    # Intentionally testing SafeTensors path if possible
    # We can try to download just the index or checking if we can find a tiny safetensors model.
    # "fxmarty/tiny-testing-gpt2-remote-code"
    # Let's try a known one: "NickleDave/tiny-bert-safetensors" (fake name)
    # Checking "HuggingFaceH4/tiny-random-Llama-3"
end
