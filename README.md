# TransformersDownloads.jl

[![Docs stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaTransformers.github.io/TransformersDownloads.jl/stable)
[![Docs dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaTransformers.github.io/TransformersDownloads.jl/dev)
[![Build Status](https://github.com/JuliaTransformers/TransformersDownloads.jl/workflows/CI/badge.svg)](https://github.com/JuliaTransformers/TransformersDownloads.jl/actions?query=workflow%3ACI+branch%3Amaster)
[![Coverage](https://codecov.io/gh/JuliaTransformers/TransformersDownloads.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaTransformers/TransformersDownloads.jl)
[![Julia 1.10+](https://img.shields.io/badge/julia-1.10%2B-blueviolet)](https://julialang.org)

⬇️ **Download and load Transformer model assets from the 🤗 Hugging Face Hub or
from local paths.**

`TransformersDownloads.jl` is the asset-loading substrate of the Julia
Transformers ecosystem ([DistilBERT.jl](https://github.com/JuliaTransformers/DistilBERT.jl)
and friends). It turns a repository id or a local path into ready-to-use Julia
data — configurations, weights and tokenizer files — and nothing more.

## ✨ Features

- 🤗 **Hugging Face Hub integration** — download and cache files by repository id.
- 🚀 **SafeTensors first** — memory-mapped, zero-copy weights, including sharded
  checkpoints (`model.safetensors.index.json`) via a lazy `ShardedStateDict`.
- 🕰️ **Legacy support** — load PyTorch Pickle (`.bin`) weights through the
  optional `Pickle.jl` extension.
- 📁 **Local paths** — point at a directory or a specific file (any name), with
  clear errors when something is missing.
- 🧩 **Extensible** — `vocab_format`, `load_vocab_file` and `load_pickle_weights`
  are documented extension points.
- 🪶 **Lightweight** — three dependencies, no model or tokenization code.

## 📦 Installation

Requires Julia 1.10 or later (package extensions).

```julia
using Pkg
Pkg.add("TransformersDownloads")
```

## ⚡ Quick start

```julia
using TransformersDownloads

# Tiny model keeps the examples light; swap in any Hugging Face repository id.
model_id = "hf-internal-testing/tiny-random-bert"

config    = load_config(model_id)      # AbstractDict
weights   = load_weights(model_id)     # AbstractDict{String,<:AbstractArray}
vocab     = load_vocab(model_id)       # Vector{String}
tokenizer = load_tokenizer(model_id)   # AbstractDict
```

## 🗂️ What you can load

| Function | Returns |
|---|---|
| `hf_file(repo_id, filename)` | local path, downloading and caching as needed |
| `hf_config`, `hf_weights`, `hf_tokenizer`, `hf_vocab`, `hf_merges` | local path of the matching asset |
| `load_config` | `config.json` as an `AbstractDict` |
| `load_weights` | state dict as `AbstractDict{String,<:AbstractArray}` |
| `load_tokenizer` | `tokenizer.json` as an `AbstractDict` |
| `load_vocab` | `Vector{String}` (`.txt`) or `AbstractDict` (`.json`) |
| `load_merges` | `merges.txt` as a `Vector{String}` |

## ⚖️ Loading weights

`load_weights` detects the available format automatically, preferring
SafeTensors over PyTorch Pickle. The returned object is always an
`AbstractDict{String, <:AbstractArray}`:

| Checkpoint | Return type | Tensor storage |
|---|---|---|
| `model.safetensors` | `SafeTensors.SafeTensor` | 🧊 memory-mapped, zero-copy |
| `model.safetensors.index.json` (sharded) | `TransformersDownloads.ShardedStateDict` | 🧊 memory-mapped, lazy |
| `pytorch_model.bin` | `Dict{String, Any}` | 📥 copied into RAM |

To load legacy `.bin` weights, load `Pickle.jl` to enable the extension:

```julia
using TransformersDownloads
using Pickle

weights = load_weights("some-legacy-model")
```

## 📁 Local paths

Every `load_*` function accepts a local directory or a direct file path in place
of a repository id:

```julia
config  = load_config("./my_model")                            # directory with config.json
weights = load_weights("./my_model")                           # directory with model.safetensors
weights = load_weights("/path/to/custom_weights.safetensors")  # direct file, any name
vocab   = load_vocab("/path/to/custom_vocab.txt")              # direct file, any name
```

Classification rules:

- an existing file or directory is used locally;
- a string starting with `/`, `.` or `~` that does not exist throws a
  `TransformersDownloads.PathNotFoundError` instead of being sent to the Hub;
- anything else is treated as a Hugging Face repository id. Prefix relative
  paths with `./` to force local treatment.

## 🚨 Error handling

All exceptions thrown by the package subtype
`TransformersDownloads.TransformersDownloadsError`, so downstream packages can
react to specific failures:

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

## 🌱 Ecosystem

`TransformersDownloads.jl` is designed to be consumed by model packages:

```julia
using TransformersDownloads

state_dict = load_weights(repo_id)   # AbstractDict{String,<:AbstractArray}
for (name, tensor) in state_dict
    # `tensor` is an AbstractArray; copy it into your model's parameters
end
```

- 🔗 Built on [`HuggingFaceApi.jl`](https://github.com/FluxML/HuggingFaceApi.jl),
  [`SafeTensors.jl`](https://github.com/FluxML/SafeTensors.jl) and
  [`JSON.jl`](https://github.com/JuliaIO/JSON.jl).
- 🧪 Designed for [DistilBERT.jl](https://github.com/JuliaTransformers/DistilBERT.jl)
  and other model packages.
- 🧭 Tokenization algorithms will live in the planned `TransformersTokenizers.jl`
  package, not here — this package only fetches and parses their files.

See the [Ecosystem Guide](https://JuliaTransformers.github.io/TransformersDownloads.jl/dev/guide/)
for the weights contract, error taxonomy and extension points.

## 📚 Documentation

Full documentation: [stable](https://JuliaTransformers.github.io/TransformersDownloads.jl/stable)
· [dev](https://JuliaTransformers.github.io/TransformersDownloads.jl/dev)

## 📄 License

MIT License
