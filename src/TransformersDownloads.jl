"""
    module TransformersDownloads

A Julia package for downloading and loading Transformer model configurations and weights
from the Hugging Face Hub or local directories.

Support for SafeTensors is built-in, while PyTorch Pickle format is supported via the
`Pickle.jl` extension.
"""
module TransformersDownloads

using HuggingFaceApi
using JSON
using SafeTensors
using Mmap

export hf_file, hf_config, hf_weights,
    load_config, load_weights

include("utils.jl")
include("hub.jl")
include("config.jl")
include("weights.jl")

end
