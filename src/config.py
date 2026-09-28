from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
def load_config(path='config.yaml'):
    with open(ROOT/path,encoding='utf-8') as f:return yaml.safe_load(f)
def ensure_dirs(cfg):
    for p in (cfg['paths']['processed'],cfg['paths']['artifacts']):(ROOT/p).mkdir(parents=True,exist_ok=True)
