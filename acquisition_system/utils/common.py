import yaml
import numpy as np

# Function to load config .yaml
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Function to generate consistent colors for each ID
def get_color_for_id(track_id):
    np.random.seed(track_id)
    return tuple(np.random.randint(0, 255, size=3).tolist())
