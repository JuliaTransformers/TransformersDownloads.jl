const CONFIG_NAME = "config.json"

"""
    hf_config(repo_id; kws...)

Download the `config.json` for a given repository ID. Returns the local path.
"""
hf_config(repo_id; kws...) = hf_file(repo_id, CONFIG_NAME; kws...)

"""
    load_config(repo_id; kws...)

Download and parse the `config.json` for a given repository ID. Returns a dictionary.
"""
load_config(repo_id; kws...) = json_load(hf_config(repo_id; kws...))
