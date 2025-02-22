import os
import argparse
from typing import *
from pathlib import Path
os.environ['HF_HOME'] = '/scratch/euijinrnd/.cache/huggingface/' # huggingface cache 를 /scratch/euijinrnd로 바꿔주기
os.environ['HF_DATASETS_OFFLINE'] = '1'

import h5py
import numpy as np
from tqdm import tqdm
import datasets
from datasets import Dataset, concatenate_datasets


def convert_dataset(task_path: Path):
    image_observations = []
    continuous_actions = []
    rewards = []
    with h5py.File(task_path, 'r') as f:
        assert 'data' in f
        for demo_name, demo in f['data'].items(): # demo iter
            actions = demo['actions']
            _obs = demo['obs']
            corner_rgb = _obs['corner_rgb'] # this is observation
            # obs_gt = obs['obs_gt']
            robot_states = _obs['robot_states'] # this is observation too.
            ep_rewards = demo['reward']
            # states = demo['states']
            # terminated = demo['terminated']
            # truncated = demo['truncated']
            
            image_observations.append(np.asarray(corner_rgb))
            continuous_actions.append(np.asarray(actions)[:, np.newaxis, :].astype(np.float32))
            rewards.append(np.asarray(ep_rewards).astype(np.float32))
        d = {
            "image_observations": image_observations, # (demo, (shape))
            "continuous_actions": continuous_actions,
            "rewards": rewards,
        }
        features = datasets.Features(
            {
                "image_observations": datasets.Sequence(datasets.Image()),
                "continuous_actions": datasets.Sequence(datasets.Array2D(shape=(1, 4), dtype='float32')),
                "rewards": datasets.Sequence(datasets.Value("float32")),
            }
        )
        ds = [
            Dataset.from_dict({k: [v[idx]] for k, v in d.items()}, features)
            for idx in range(len(d["image_observations"]))
        ]
        return concatenate_datasets(ds)


def main(args):
    for task_path in tqdm(list(args.data_path.glob('*.hdf5'))):
        dataset = convert_dataset(task_path)
        dataset.save_to_disk(args.output_path.joinpath(task_path.stem))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=Path, default=Path('data/metaworld/ML45/train'))
    parser.add_argument('--output-path', type=Path, default=Path('/home/euijinrnd/workspace/jat/converted_data/metaworld'))
    args = parser.parse_args()
    main(args)