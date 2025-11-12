import yaml, os

def load_config(path: str):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    # Resolve paths relative to project root (file location two dirs up from here)
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    for k in ['checkpoint_dir', 'replay_dir']:
        p = cfg['paths'][k]
        if not os.path.isabs(p):
            cfg['paths'][k] = os.path.join(root, p)
    return cfg
