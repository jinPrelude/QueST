from concurrent.futures import ProcessPoolExecutor

import os
import argparse
from typing import Tuple
from pathlib import Path

os.environ['HF_HOME'] = '/scratch/euijinrnd/.cache/huggingface/' # huggingface cache 를 /scratch/euijinrnd로 바꿔주기
os.environ['HF_DATASETS_OFFLINE'] = '1'

import h5py
import numpy as np
import cv2  # using OpenCV for fast image resizing
from tqdm import tqdm
import datasets
from datasets import Dataset

def resize_image(image: np.ndarray, size: Tuple[int, int] = (84, 84), normalize: bool = False) -> np.ndarray:
    # cv2.resize expects size as (width, height)
    resized = cv2.resize(image, (size[0], size[1]), interpolation=cv2.INTER_CUBIC)
    if normalize:
        # Convert image to float32 and normalize pixel values to [0, 1]
        resized = resized.astype(np.float32) / 255.0
    return resized

def convert_dataset(task_path: Path, normalize: bool = False):
    image_observations = []
    continuous_actions = []
    rewards = []
    with h5py.File(task_path, 'r') as f:

        for demo in f['data'].values():
            _obs = demo['obs']
            # Assume corner_rgb has shape (seq_len, 128, 128, 3)
            original_images = np.asarray(_obs['corner_rgb'])
            # Resize each frame in the sequence
            resized_images = np.array([
                resize_image(frame, size=(84, 84), normalize=normalize) 
                for frame in original_images
            ])
            image_observations.append(resized_images)
            
            continuous_actions.append(
                np.asarray(demo['actions'])[:, np.newaxis, :].astype(np.float32)
            )
            rewards.append(
                np.asarray(demo['reward']).astype(np.float32)
            )
            
        d = {
            "image_observations": image_observations,
            "continuous_actions": continuous_actions,
            "rewards": rewards,
        }
        features = datasets.Features({
            "image_observations": datasets.Sequence(datasets.Image()),
            "continuous_actions": datasets.Sequence(datasets.Array2D(shape=(1, 4), dtype='float32')),
            "rewards": datasets.Sequence(datasets.Value("float32")),
        })
        # features = datasets.Features({
        #     "image_observations": datasets.Sequence(datasets.Array3D(shape=(84, 84, 3), dtype='float32')),
        #     "continuous_actions": datasets.Sequence(datasets.Array2D(shape=(1, 4), dtype='float32')),
        #     "rewards": datasets.Sequence(datasets.Value("float32")),
        # })
        return Dataset.from_dict(d, features=features)


def process_file(task_path: Path, output_dir: Path):
    output_path = output_dir.joinpath(task_path.stem)
    if output_path.exists():
        print(f"{output_path} exists. skipped.")
        return
    dataset = convert_dataset(task_path)
    dataset.save_to_disk(output_path)

def main(args):
    task_paths = list(args.data_path.glob('*.hdf5'))
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_file, tp, args.output_dir) for tp in task_paths]
        for future in tqdm(futures):
            future.result()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=Path, default=Path('data/metaworld/ML45/train'))
    parser.add_argument('--output-dir', type=Path, default=Path('/home/euijinrnd/workspace/jat/converted_data/metaworld_array'))
    parser.add_argument('--normalize', action='store_true', help="Normalize images to range [0, 1]")
    args = parser.parse_args()
    main(args)
