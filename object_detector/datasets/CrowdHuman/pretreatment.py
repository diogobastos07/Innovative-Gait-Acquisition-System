import yaml
import os
from pathlib import Path
import zipfile
import json
from PIL import Image
import tqdm
from multiprocessing import Pool
from functools import partial
import shutil



def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config



def process_image_with_annotation(annotation_data, output_path, filter_cfg, aux):
    """Normalizes annotations, filters those that are full_body (fbox), 
    and those fbox that do not go beyond the image limits and respect the minimum occlusion threshold.
    """

    id = annotation_data.get("ID")
    image = Image.open(os.path.join(output_path, 'intermediate_stage', 'Images', id + '.jpg'))
    width, height = image.size
    annotation_data_normalized = []
    for obj in annotation_data['gtboxes']:
        if obj['tag'] == 'person':
            x_min_f, y_min_f, width_f, height_f = obj['fbox']
            _, _, width_v, height_v = obj['vbox']
            if (x_min_f >= 0) and (y_min_f >= 0) and (x_min_f + width_f <= width) and (y_min_f + height_f <= height):
                if (width_f * height_f * filter_cfg['max_occlusion_ratio']) <= (width_v * height_v):
                    x_center_norm = "{:.5f}".format((x_min_f + (width_f / 2)) / width)
                    y_center_norm = "{:.5f}".format((y_min_f + (height_f / 2)) / height)
                    width_norm =  "{:.5f}".format(width_f / width)
                    height_norm = "{:.5f}".format(height_f / height)
                    annotation_data_normalized.append('0 ' + str(x_center_norm) + ' ' + str(y_center_norm) + ' ' + str(width_norm) + ' ' + str(height_norm))
    if annotation_data_normalized or filter_cfg['keep_images_without_annotations']:
        set_Data = 'Validation' if aux else 'Train'
        with open(os.path.join(output_path, 'labels', set_Data, id + '.txt'), 'w') as txt_file:
            for box in annotation_data_normalized:
                txt_file.write(f'{box}\n')
        shutil.move(os.path.join(output_path, 'intermediate_stage', 'Images', id + '.jpg'), os.path.join(output_path, 'images', set_Data))



def transform(base_path: Path, output_path: Path, filter_cfg: dict, workers=1) -> None:
    """Filters images and annotations from base_path to output_path using parallel workers.
    Args:
        base_path (Path): Path to the directory containing the archives.
        output_path (Path): Path to the directory to extract the archives to.
        filter_cfg (dict): Configuration for filtering images/annotations.
        workers (int): Number of parallel workers to use.
    """

    directories = ['images/Train', 'images/Validation', 'labels/Train', 'labels/Validation', 'intermediate_stage']
    os.makedirs(output_path, exist_ok=True)
    for directory in directories:
        os.makedirs(os.path.join(output_path, directory), exist_ok=True)

    folder_names = [['CrowdHuman_train01.zip', 'CrowdHuman_train02.zip', 'CrowdHuman_train03.zip'], ['CrowdHuman_val.zip']]
    annotations_file_names = ['annotation_train.odgt', 'annotation_val.odgt']

    for i, set_Data in enumerate(folder_names):
        print('Reading ' + annotations_file_names[i] + '...')
        annotations_data = []
        with open(os.path.join(base_path, annotations_file_names[i]), "r") as annotations_file_name:
            for line in annotations_file_name:
                line_json = json.loads(line)
                annotations_data.append(line_json)
        for folder in set_Data:
            print('Unzipping ' + folder + '...')
            with open(os.path.join(base_path, folder), 'rb') as f:
                z = zipfile.ZipFile(f)
                z.extractall(os.path.join(output_path, 'intermediate_stage'))
            ids = os.listdir(os.path.join(output_path, 'intermediate_stage', 'Images'))
            print('Preprocessing ' + folder + '...')
            tasks = []
            for id in ids:
                annotation_data = next((ann for ann in annotations_data if (ann.get('ID') + '.jpg') == id), None)
                if annotation_data:
                    tasks.append(annotation_data)   
            progress = tqdm.tqdm(total=len(ids), desc='Preprocessing', unit='img')
            with Pool(processes=workers) as pool:
                for _ in pool.imap(partial(process_image_with_annotation, output_path = output_path, filter_cfg = filter_cfg, aux = i), tasks):
                    progress.update(1)
            progress.close()
            shutil.rmtree(os.path.join(output_path, 'intermediate_stage', 'Images'))
    
    num_train_img = len(os.listdir(os.path.join(output_path, 'images', 'Train')))
    num_valid_img = len(os.listdir(os.path.join(output_path, 'images', 'Validation')))

    print("\nFILTERING SUMMARY:")
    print("-" * 70)
    print(f'| Using: - max_occlusion_ratio = {filter_cfg["max_occlusion_ratio"]} (visibel_area / full_area)')
    print(f'|        - keep_images_without_annotations = {filter_cfg["keep_images_without_annotations"]}')
    print("-" * 70)
    print(f"| Initial number of images in training set: {15000}")
    print(f"| Initial number of images in validation set: {4370}")
    print("-" * 70)
    print(f"| Final number of images in training set: {num_train_img}")
    print(f"| Final number of images in validation set: {num_valid_img}")
    print("-" * 70)

    os.rmdir(os.path.join(output_path, 'intermediate_stage'))    



if __name__ == '__main__':
    # Load config file .yaml
    current_directory = os.path.dirname(os.path.abspath(__file__)) 
    cfg_path = os.path.join(current_directory, 'config.yaml')
    print(f"Loading {cfg_path} ...")
    cfg = load_config(cfg_path)

    data_cfg = cfg['data_cfg']
    filter_cfg = cfg['filtering_cfg']
    print('\n------------------------------- ' + data_cfg['dataset_name'] + ' DATASET -------------------------------\n')
    transform(Path(data_cfg['dataset_input_root']), Path(data_cfg['dataset_output_root']), filter_cfg, data_cfg['num_workers'])
    print('\n-------------------- ' + data_cfg['dataset_name'] + ' DATASET PRETREATMENT COMPLETE! --------------------\n')
    