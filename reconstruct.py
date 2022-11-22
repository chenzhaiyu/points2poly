"""
Reconstruct surface models from point clouds.

The pipeline consists of three components: (a) a cell complex is generated
 via adaptive space partitioning that provides a polyhedral embedding;
 (b) an implicit field is learned by a deep neural network that facilitates
 building occupancy estimation; (c) a Markov random field is formulated
 to extract the outer surface of a building via combinatorial optimisation.

To reconstruct the surface model, one needs to first train the points2surf
 neural network on a representative dataset (such as the provided Helsinki buildings).
 The code below assumes such training has been done therefore object occupancy
 shall be inferred.
"""

import glob
import numpy as np
from pathlib import Path
import pickle
import os
import sys

# intact points2surf submodule
sys.path.append(os.path.abspath('points2surf'))

import hydra
from omegaconf import DictConfig

from utils import create_cell_complex, create_query_points, extract_surface, infer_sdf


@hydra.main(config_path='./conf', config_name='config', version_base='1.2')
def reconstruct_full(cfg: DictConfig):
    """
    Full reconstruction pipeline starting from point clouds.

    Parameters
    ----------
    cfg: DictConfig
        Hydra configuration
    """

    # create cell complexes and save query points (.npy)
    complexes = {}
    for filepath in glob.glob(cfg.dataset_paths):
        filepath = Path(filepath)

        # prioritise_verticals for buildings
        # normalise_vg for vg file in arbitrary scale (not created from normalised point cloud)
        # filepath_write_candidate = filepath.with_suffix('.plm')
        cell_complex = create_cell_complex(filepath,
                                           filepath_out=None, theta=cfg.refine_theta, epsilon=cfg.refine_epsilon,
                                           prioritise_verticals=cfg.prioritise_verticals,
                                           normalise=cfg.normalise,
                                           append_bottom=cfg.append_bottom,)  # only for MVS point cloud without bottom

        complexes.update({filepath.stem: cell_complex})
        create_query_points(cell_complex,
                            filepath_query=(filepath.parent.parent / '05_query_pts' / filepath.name).with_suffix(
                                '.ply.npy'),
                            filepath_dist=(filepath.parent.parent / '05_query_dist' / filepath.name).with_suffix(
                                '.ply.npy'))

    # dump complexes
    if cfg.complexes_path:
        Path(cfg.complexes_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cfg.complexes_path, 'wb') as f_complexes:
            pickle.dump(complexes, f_complexes)

    # batch prediction and save sdf values (.npy)
    infer_sdf(cfg)

    # extract surfaces (.obj)
    for name in complexes:
        sdf_path = (Path(cfg.outdir) / 'eval/eval' / name).with_suffix('.xyz.npy')
        if not sdf_path.exists():
            print('skipping {}'.format(sdf_path))
            continue

        sdf_values = np.load(sdf_path)
        extract_surface((Path(cfg.result_dir) / name).with_suffix('.obj'), complexes[name],
                        sdf_values, graph_cut=True, coefficient=cfg.coefficient)


@hydra.main(config_path='./conf', config_name='config', version_base='1.2')
def reconstruct_surface(cfg: DictConfig):
    """
    Reconstruction of surface extraction from cell complexes and SDF predictions.

    Parameters
    ----------
    cfg: DictConfig
        Hydra configuration
    """

    # load cell complexes
    with open(cfg.complexes_path, 'rb') as f_complexes:
        complexes = pickle.load(f_complexes)
        print('{} loaded'.format(cfg.complexes_path))

    for name in complexes:
        # load prediction results
        sdf_path = (Path(cfg.outdir) / 'eval/eval' / name).with_suffix('.xyz.npy')
        sdf_values = np.load(sdf_path)

        # surface extraction
        extract_surface((Path(cfg.result_dir) / name).with_suffix('.obj'), complexes[name],
                        sdf_values, graph_cut=True, coefficient=cfg.coefficient)


if __name__ == '__main__':
    reconstruct_full()
