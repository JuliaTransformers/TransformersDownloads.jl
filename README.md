# TransformersDownloads.jl

[![Docs stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaTransformers.github.io/TransformersDownloads.jl/stable)
[![Docs dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaTransformers.github.io/TransformersDownloads.jl/dev)
[![Build Status](https://github.com/JuliaTransformers/TransformersDownloads.jl/workflows/CI/badge.svg)](https://github.com/JuliaTransformers/TransformersDownloads.jl/actions?query=workflow%3ACI+branch%3Amaster)
[![Coverage](https://codecov.io/gh/JuliaTransformers/TransformersDownloads.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaTransformers/TransformersDownloads.jl)
[![Julia 1.10+](https://img.shields.io/badge/julia-1.10%2B-blueviolet)](https://julialang.org)

`TransformersDownloads.jl` downloads model files from the Hugging Face Hub, or
reads them from a local directory or file, and returns them as Julia data. It
handles the files needed to run a model: `config.json`, SafeTensors and PyTorch
weight files, `tokenizer.json`, vocabularies and `merges.txt`.

The package contains no model or tokenization code. Model architectures live in
packages such as [DistilBERT.jl](https://github.com/JuliaTransformers/DistilBERT.jl),
and tokenization algorithms are planned for a separate
`TransformersTokenizers.jl` package.

## Features

- Fetch any file from a repository by id with `hf_file`; downloads are cached
  by [HuggingFaceApi.jl](https://github.com/FluxML/HuggingFaceApi.jl).
- Load SafeTensors checkpoints, single-file or sharded, without copying tensor
  data into memory.
- Load legacy PyTorch `.bin` checkpoints through the optional
  [Pickle.jl](https://github.com/chengchingwen/Pickle.jl) extension.
- Read the same assets from a local directory or file by passing a path instead
  of a repository id.
- Add support for other vocabulary formats through `vocab_format` and
  `load_vocab_file`.

## Installation

Julia 1.10 or later is required (package extensions).

```julia
using Pkg
Pkg.add("TransformersDownloads")
```

## Usage

```julia
using TransformersDownloads

# Any repository id works; this tiny model keeps the example quick.
model_id = "hf-internal-testing/tiny-random-bert"

config    = load_config(model_id)       # parsed config.json
weights   = load_weights(model_id)      # parameter names => arrays
vocab     = load_vocab(model_id)        # Vector{String} for vocab.txt
tokenizer = load_tokenizer(model_id)    # parsed tokenizer.json
```

Each `load_*` function also accepts a local directory or file:

```julia
config  = load_config("./my_model")
weights = load_weights("/path/to/custom_weights.safetensors")
```

## Functions

| Function | Returns |
|---|---|
| `hf_file(repo_id, filename)` | local path of any file, downloaded and cached if needed |
| `hf_config`, `hf_weights`, `hf_tokenizer`, `hf_vocab`, `hf_merges` | local path of that asset |
| `load_config` | `config.json` parsed into a dict |
| `load_weights` | parameter names mapped to arrays |
| `load_tokenizer` | `tokenizer.json` parsed into a dict |
| `load_vocab` | `Vector{String}` for `.txt`, dict for `.json` |
| `load_merges` | `Vector{String}` |

## Weights

`load_weights` picks the format for you:

- `model.safetensors` returns a `SafeTensors.SafeTensor`. Tensor data stays
  memory-mapped and is only read when accessed.
- `model.safetensors.index.json` returns a `ShardedStateDict`, a lazy view over
  the shard files.
- `pytorch_model.bin` returns a `Dict{String, Any}`, copied into memory.

All three satisfy `AbstractDict{String, <:AbstractArray}`, so the same loop
works for any format:

```julia
for (name, tensor) in load_weights(model_id)
    # tensor is an AbstractArray
end
```

To read `.bin` files, load Pickle.jl to enable the extension:

```julia
using TransformersDownloads
using Pickle

weights = load_weights("some-model-with-bin-weights")
```

## Local files

If the first argument is an existing file or directory, it is used directly
instead of being treated as a repository id. This makes it possible to work
with downloaded or converted checkpoints without touching the Hub:

```julia
config  = load_config("./my_model")                            # directory with config.json
weights = load_weights("./my_model")                           # directory with model.safetensors
weights = load_weights("/path/to/custom_weights.safetensors")  # direct file
vocab   = load_vocab("/path/to/custom_vocab.txt")              # direct file
```

A path starting with `/`, `.` or `~` that does not exist throws a
`PathNotFoundError` instead of being sent to the Hub as a repository id.
Relative paths that do not start with `.` are treated as repository ids; write
`./my_model` to force the local interpretation.

## Errors

All exceptions subtype `TransformersDownloads.TransformersDownloadsError`, so
callers can react to specific failures:

| Exception | Meaning |
|---|---|
| `PathNotFoundError` | a path-like string does not exist locally |
| `LocalFileNotFoundError` | a directory does not contain the requested file |
| `WeightFormatNotFoundError` | no supported weights found |
| `VocabFileNotFoundError` | neither `vocab.txt` nor `vocab.json` found |
| `UnsupportedVocabFormatError` | a vocabulary file exists but its format is unknown |
| `PickleSupportNotLoadedError` | `.bin` weights requested without `Pickle.jl` |

For example, a model package can fall back to random initialization when no
weights are available:

```julia
using TransformersDownloads: TransformersDownloadsError, WeightFormatNotFoundError

try
    load_weights("./my_model")
catch err
    if err isa WeightFormatNotFoundError
        @warn "no weights found; initializing randomly" err.files
    else
        rethrow()
    end
end
```

## Documentation

The [Ecosystem Guide](https://JuliaTransformers.github.io/TransformersDownloads.jl/dev/guide/)
describes the weights contract and the extension points for package authors.
Full API documentation is available for the
[stable](https://JuliaTransformers.github.io/TransformersDownloads.jl/stable)
and [dev](https://JuliaTransformers.github.io/TransformersDownloads.jl/dev)
versions.

## License

MIT License
