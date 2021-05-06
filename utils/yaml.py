import yaml
from collections import defaultdict
import os
def read_yaml(path):
    if os.path.isfile(path):
        data = yaml.safe_load(open(path))
        return defaultdict(dict, {} if data is None else data)
    else:
        return defaultdict(dict, {})

def save_yaml(path, data):
    yaml_file = open(path, "w+")
    yaml.dump(dict(data), yaml_file)
    yaml_file.close()
