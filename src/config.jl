const CONFIG_NAME = "config.json"

"""
    hgf_model_config(model_name; kws...)

Download the `config.json` for a given model.
"""
hgf_model_config(model_name; kws...) = hgf_file(model_name, CONFIG_NAME; kws...)

"""
    load_config(model_name; kws...)

Download and parse the `config.json` for a given model.
"""
load_config(model_name; kws...) = json_load(hgf_model_config(model_name; kws...))
