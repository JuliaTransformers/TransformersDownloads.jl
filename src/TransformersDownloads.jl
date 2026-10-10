"""
    module TransformersDownloads

A Julia package for downloading and loading Transformer model configurations and
weights from the Hugging Face Hub or local directories.

Support for SafeTensors is built-in, while PyTorch Pickle format is supported
via the `Pickle.jl` extension.
"""
module TransformersDownloads

using HuggingFaceApi
using HuggingFaceApi: list_model_files, get_token
using JSON
using SafeTensors

export hf_file, hf_config, hf_weights, hf_tokenizer, hf_vocab, hf_merges,
    load_config, load_weights, load_tokenizer, load_vocab, load_merges

include("errors.jl")
include("sources.jl")
include("utils.jl")
include("hub.jl")
include("config.jl")
include("weights.jl")
include("tokenizer.jl")

end
