import yaml
from collections import defaultdict


def read_yaml(path):
    data = yaml.safe_load(open(path))
    return defaultdict(dict, {} if data is None else data)

def save_yaml(path, data):
    yaml_file = open("params.yaml", "r+")
    yaml.dump(dict(data), yaml_file)
    yaml_file.close()
