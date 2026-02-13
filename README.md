# TransformersDownloads.jl

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaLang.github.io/TransformersDownloads.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaLang.github.io/TransformersDownloads.jl/dev)
[![Build Status](https://github.com/JuliaTransformers/TransformersDownloads.jl/workflows/CI/badge.svg)](https://github.com/JuliaTransformers/TransformersDownloads.jl/actions?query=workflow%3ACI+branch%3Amaster)

`TransformersDownloads.jl` is a Julia package for downloading and loading Transformer model configurations and weights from the Hugging Face Hub (or local directories).

It focuses on providing a lightweight, efficient, and consistent interface for accessing model assets, with a strong emphasis on `SafeTensors` and lazy loading via `Mmap`.

## Features

- **Hugging Face Hub Integration**: Easily download files, configurations, and weights using repo IDs.
- **SafeTensors Support**: Native support for `.safetensors` files, including sharded weights (`.safetensors.index.json`).
- **Legacy Support**: Load PyTorch Pickle weights (`.bin`) via the optional `Pickle.jl` extension.
- **Lazy Loading**: Uses `Mmap` and `SafeTensors` to load weights efficiently without necessarily loading everything into RAM at once.
- **Local Fallback**: If a path is provided instead of a repo ID, it automatically loads from the local directory.

## Installation

```julia
using Pkg
Pkg.add("TransformersDownloads")
```

## Usage

### Downloading Files

You can download any file from a Hugging Face repository:

```julia
using TransformersDownloads

# Download a specific file
path = hf_file("google-bert/bert-base-uncased", "tokenizer.json")
```

### Loading Configuration

Load the `config.json` of a model as a Julia dictionary:

```julia
config = load_config("google-bert/bert-base-uncased")
# Returns a Dict{String, Any}
```

### Loading Weights

Load the weights (state dictionary) of a model. The library will automatically detect the format (SafeTensors is preferred).

```julia
# This will look for model.safetensors or pytorch_model.bin
weights = load_weights("google-bert/bert-base-uncased")
```

#### SafeTensors (Recommended)
If the model has `.safetensors` files, they will be loaded using `SafeTensors.jl`. This is efficient and secure.

#### PyTorch Pickle Support
To load legacy `.bin` weights, you must have `Pickle.jl` loaded in your environment:

```julia
using TransformersDownloads
using Pickle # Enables Pickle support via extension

# Now you can load .bin weights
weights = load_weights("some-legacy-model")
```

### Using Local Directories

If you have a model downloaded locally, just pass the directory path:

```julia
# If "./my_model/" contains config.json and model.safetensors
config = load_config("./my_model/")
weights = load_weights("./my_model/")
```

## API Reference

- `hf_file(repo_id, filename; revision="main", auth_token=...)`: Download a file.
- `hf_config(repo_id; ...)`: Download and get the path to `config.json`.
- `hf_weights(repo_id; ...)`: Download and get the path to weights.
- `load_config(repo_id; ...)`: Download and parse `config.json`.
- `load_weights(repo_id; ...)`: Download and load weights into a dictionary-like object.

## License

MIT License
