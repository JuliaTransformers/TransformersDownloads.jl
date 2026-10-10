const CONFIG_NAME = "config.json"

"""
    hf_config(repo_id; kws...)

Download the `config.json` for a given repository ID or resolve it from a local
path. Returns the local path. See [`hf_file`](@ref) for the accepted sources and
keyword arguments.
"""
hf_config(repo_id; kws...) = hf_file(repo_id, CONFIG_NAME; kws...)

"""
    load_config(repo_id; kws...)

Download and parse the `config.json` for a given repository ID or local path.
Returns a dictionary. See [`hf_file`](@ref) for the accepted sources and keyword
arguments.
"""
load_config(repo_id; kws...) = json_load(hf_config(repo_id; kws...))
