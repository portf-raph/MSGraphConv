# Courtesy of https://github.com/lrjconan/LanczosNetwork/blob/master/utils/arg_helper.py
# (MIT Licensed source)


import os
import yaml
import time
from easydict import EasyDict as edict


def get_config(config_file, exp_dir=None):
    """ Construct and snapshot hyper parameters """
    script_cfg = edict(yaml.safe_load(open(config_file, 'r')))    # safe_load (mod)

    # create hyper parameters
    script_cfg.run_id = str(os.getpid())
    script_cfg.exp_name = '_'.join([
        script_cfg.model.name, script_cfg.dataset.name,
        time.strftime('%Y-%b-%d-%H-%M-%S'), script_cfg.run_id
    ])

    if exp_dir is not None:
        script_cfg.exp_dir = exp_dir

    script_cfg.save_dir = os.path.join(script_cfg.exp_dir, script_cfg.exp_name)

    # snapshot hyperparameters
    mkdir(script_cfg.exp_dir)
    mkdir(script_cfg.save_dir)

    save_name = os.path.join(script_cfg.save_dir, 'script_cfg.yaml')
    yaml.dump(edict2dict(script_cfg), open(save_name, 'w'), default_flow_style=False)

    return script_cfg


def get_config_(config_file, exp_dir=None):
    cfg = edict(yaml.safe_load(open(config_file, 'r')))   # safe_load (mod)

    return cfg


def edict2dict(edict_obj):
    dict_obj = {}

    for key, vals in edict_obj.items():
        if isinstance(vals, edict):
            dict_obj[key] = edict2dict(vals)
        else:
            dict_obj[key] = vals

    return dict_obj


def mkdir(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)
