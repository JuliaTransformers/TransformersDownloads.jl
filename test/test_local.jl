using Test
using JSON
using SafeTensors
using TransformersDownloads

if !isdefined(@__MODULE__, :TD)
    const TD = TransformersDownloads
end

function build_local_fixtures(root)
    single = joinpath(root, "single")
    mkpath(single)
    a = Float32[1 2; 3 4]
    b = Float64[1.0, 2.0, 3.0]
    SafeTensors.serialize(joinpath(single, "model.safetensors"), Dict("a" => a, "b" => b))
    write(joinpath(single, "config.json"),
          JSON.json(Dict("hidden_size" => 4, "num_hidden_layers" => 2)))
    write(joinpath(single, "vocab.txt"), "[PAD]\n[UNK]\n[CLS]\n[SEP]\n")
    write(joinpath(single, "merges.txt"), "#version: 0.2\na b\n")
    write(joinpath(single, "tokenizer.json"), JSON.json(Dict("model" => Dict("type" => "WordPiece"))))

    custom_weights = joinpath(root, "my_weights.safetensors")
    SafeTensors.serialize(custom_weights, Dict("a" => a))

    custom_vocab = joinpath(root, "my_vocab.txt")
    write(custom_vocab, "[PAD]\n[UNK]\n[CLS]\n")

    custom_config = joinpath(root, "my_config.json")
    write(custom_config, JSON.json(Dict("hidden_size" => 4)))

    json_vocab_dir = joinpath(root, "json_vocab")
    mkpath(json_vocab_dir)
    write(joinpath(json_vocab_dir, "vocab.json"), JSON.json(Dict("hello" => 0, "world" => 1)))

    empty_dir = joinpath(root, "empty")
    mkpath(empty_dir)

    return (; single, a, b, custom_weights, custom_vocab, custom_config,
              json_vocab_dir, empty_dir)
end

function build_sharded_fixture(root)
    dir = joinpath(root, "sharded")
    mkpath(dir)
    a = Float32[1 2; 3 4]
    b = Float64[5.0, 6.0, 7.0]
    SafeTensors.serialize(joinpath(dir, "model-00001-of-00002.safetensors"), Dict("a" => a))
    SafeTensors.serialize(joinpath(dir, "model-00002-of-00002.safetensors"), Dict("b" => b))
    write(joinpath(dir, "model.safetensors.index.json"), JSON.json(Dict(
        "metadata" => Dict{String,Any}(),
        "weight_map" => Dict(
            "a" => "model-00001-of-00002.safetensors",
            "b" => "model-00002-of-00002.safetensors",
        ),
    )))
    return (; dir, a, b)
end

@testset "local paths" begin
    root = mktempdir()
    fx = build_local_fixtures(root)
    sh = build_sharded_fixture(root)

    @testset "source classification" begin
        @test TD.classify_source(fx.single) isa TD.PathSource
        @test TD.classify_source(joinpath(fx.single, "model.safetensors")) isa TD.PathSource
        @test TD.classify_source("google-bert/bert-base-uncased") isa TD.RepoSource
        @test TD.classify_source("gpt2") isa TD.RepoSource
        @test_throws TD.PathNotFoundError TD.classify_source("~/definitely-not-here-xyz")
        @test_throws TD.PathNotFoundError TD.classify_source(joinpath(root, "nope"))
        @test_throws TD.PathNotFoundError TD.classify_source("./nope")
        @test TD.is_path_like("/abs/path")
        @test TD.is_path_like("./rel")
        @test TD.is_path_like("~/home")
        @test !TD.is_path_like("org/name")
    end

    @testset "hf_file" begin
        @test hf_file(fx.single, "config.json") == joinpath(fx.single, "config.json")
        @test hf_file(joinpath(fx.single, "model.safetensors"), "anything.bin") ==
              joinpath(fx.single, "model.safetensors")
        @test_throws TD.LocalFileNotFoundError hf_file(fx.single, "missing.json")
        @test_throws TD.PathNotFoundError hf_file(joinpath(root, "nope"), "config.json")
        @test TD.hf_file_url("org/name", "config.json").revision == "main"
    end

    @testset "config" begin
        config = load_config(fx.single)
        @test config isa AbstractDict
        @test config["hidden_size"] == 4
        @test load_config(fx.custom_config)["hidden_size"] == 4
        @test_throws TD.LocalFileNotFoundError load_config(fx.empty_dir)
        @test_throws TD.PathNotFoundError load_config(joinpath(root, "nope"))
    end

    @testset "weights (single safetensors)" begin
        weights = load_weights(fx.single)
        @test weights isa SafeTensors.SafeTensor
        @test weights isa AbstractDict{String,<:AbstractArray}
        @test Set(keys(weights)) == Set(["a", "b"])
        @test weights["a"] == fx.a
        @test weights["b"] == fx.b

        weights2 = load_weights(joinpath(fx.single, "model.safetensors"))
        @test weights2["a"] == fx.a

        weights3 = load_weights(fx.custom_weights)
        @test weights3 isa SafeTensors.SafeTensor
        @test weights3["a"] == fx.a
    end

    @testset "weights (sharded safetensors)" begin
        weights = load_weights(sh.dir)
        @test weights isa TD.ShardedStateDict
        @test weights isa AbstractDict{String,<:AbstractArray}
        @test length(weights) == 2
        @test Set(keys(weights)) == Set(["a", "b"])
        @test haskey(weights, "a")
        @test !haskey(weights, "zzz")
        @test weights["a"] == sh.a
        @test weights["b"] == sh.b

        as_dict = Dict(weights)
        @test as_dict["a"] == sh.a
        @test as_dict["b"] == sh.b
        @test length(collect(values(weights))) == 2
    end

    @testset "weights errors" begin
        err = try
            load_weights(fx.empty_dir)
            nothing
        catch e
            e
        end
        @test err isa TD.WeightFormatNotFoundError
        @test occursin("no supported weight file", sprint(showerror, err))
        @test_throws TD.PathNotFoundError load_weights(joinpath(root, "nope"))
        @test_throws ArgumentError load_weights(fx.single; revison="x")
        @test_throws TD.LocalFileNotFoundError load_weights(fx.empty_dir;
            possible_files=["model.safetensors"])
    end

    @testset "vocab" begin
        vocab = load_vocab(fx.single)
        @test vocab isa Vector{String}
        @test "[CLS]" in vocab
        @test hf_vocab(fx.single) == joinpath(fx.single, "vocab.txt")

        vocab_json = load_vocab(fx.json_vocab_dir)
        @test vocab_json isa AbstractDict
        @test vocab_json["hello"] == 0

        vocab_custom = load_vocab(fx.custom_vocab)
        @test vocab_custom isa Vector{String}

        @test_throws TD.VocabFileNotFoundError load_vocab(fx.empty_dir)
        @test_throws ArgumentError load_vocab(fx.single; revison="x")
    end

    @testset "merges and tokenizer" begin
        @test load_merges(fx.single) == ["#version: 0.2", "a b"]
        tokenizer = load_tokenizer(fx.single)
        @test tokenizer isa AbstractDict
        @test haskey(tokenizer, "model")
        @test hf_tokenizer(fx.single) == joinpath(fx.single, "tokenizer.json")
    end
end
