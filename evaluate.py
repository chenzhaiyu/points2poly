"""
Evaluate reconstruction results by Hausdorff distance.
"""

import os
import sys

# intact points2surf submodule
sys.path.append(os.path.abspath('points2surf'))

import hydra
from omegaconf import DictConfig

from points2surf.source.base import evaluation


@hydra.main(config_path='./conf', config_name='config', version_base='1.2')
def evaluate_hausdorff_dist(cfg: DictConfig):
    """
    Evaluate Hausdorff distance between reconstructed and GT models.

    Parameters
    ----------
    cfg: DictConfig
        Hydra configuration
    """

    csv_file = os.path.join(cfg.result_dir, cfg.eval_csv)
    evaluation.mesh_comparison(
        new_meshes_dir_abs=cfg.result_dir,
        ref_meshes_dir_abs=cfg.gt_dir,
        num_processes=cfg.workers,
        report_name=csv_file,
        samples_per_model=cfg.eval_samples,
        dataset_file_abs=cfg.dataset)


if __name__ == '__main__':
    evaluate_hausdorff_dist()
