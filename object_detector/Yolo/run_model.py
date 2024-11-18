import argparse
from ultralytics import YOLO
import os
import yaml


parser = argparse.ArgumentParser(description='Main program for object detector.')
parser.add_argument('--phase', default='train', choices=['train', 'test'], help="choose train or test phase")
opt = parser.parse_args()



def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config



if __name__ == '__main__':
    cfg_path = os.path.abspath('object_detector/Yolo/Yolo.yaml')
    print(f"Loading {cfg_path} ...")
    cfg = load_config(cfg_path)

    training = (opt.phase == 'train')
    if training:
        print('\n------------------------------- TRAIN -------------------------------\n')
        YOLO().train(**cfg['train_cfg'])
    else:
        print('\n------------------------------- EVALUATION -------------------------------\n')
        model = YOLO(os.path.join(cfg['train_cfg']['project'], cfg['train_cfg']['name'], 'weights', 'best.pt'))
        model.val(**cfg['val_cfg'])