"""
Piece-wise planar reconstruction.

"""
import os
import glob
import numpy as np
from pathlib import Path

from cellcomplex import VertexGroup, CellComplex, get_logger

from source import points_to_surf_eval
from source.base import evaluation
from source import sdf

logger = get_logger(filepath='reconstruction.log')


def evaluate(opt=None):
    if not opt:
        class OPT:
            pass

        opt = OPT
        opt.seed = 40938661
        opt.indir = 'datasets'
        opt.outdir = 'results'
        opt.modeldir = 'models'
        opt.dataset = 'helsinki_noise_free/testset.txt'
        opt.models = 'p2s_max'
        opt.modelpostfix = '_model_249.pth'
        opt.batchSize = 101
        opt.workers = 1
        opt.cache_capacity = 1
        opt.query_grid_resolution = 256
        opt.epsilon = 3
        opt.certainty_threshold = 13
        opt.sigma = 5
        opt.gpu_idx = 0
        opt.parampostfix = '_params.pth'
        opt.sampling = 'full'

    indir_root = opt.indir
    outdir_root = os.path.join(opt.outdir, opt.models + os.path.splitext(opt.modelpostfix)[0])
    datasets = opt.dataset
    if not isinstance(datasets, list):
        datasets = [datasets]
    for dataset in datasets:
        print(f'Evaluating on dataset {dataset}')
        opt.indir = os.path.join(indir_root, os.path.dirname(dataset))
        opt.outdir = os.path.join(outdir_root, os.path.dirname(dataset))
        opt.dataset = os.path.basename(dataset)

        # evaluate
        opt.reconstruction = False
        points_to_surf_eval.points_to_surf_eval(opt)


def reconstruct(data_paths):
    complexes = {}
    for filename in glob.glob(data_paths):
        # load planes and bounds from vg data of a (complete) point cloud
        vertex_group = VertexGroup(filename)
        planes, bounds = np.array(vertex_group.planes), np.array(vertex_group.bounds)

        # construct cell complex and extract the cell centers as query points
        cell_complex = CellComplex(planes, bounds)
        cell_complex.prioritise_planes()
        cell_complex.construct()
        # cell_complex.visualise()
        cell_complex.save_plm(
            (Path('results') / 'p2s_max_model_249' / 'helsinki_noise_free/eval/ply_candidates' / Path(filename).name).with_suffix('.plm'), polygonal=True)
        cell_complex.print_info()
        queries = np.array(cell_complex.cell_centers(mode='centroid'), dtype=np.float32)

        # save the query points to 05_query_pts
        save_path = (Path(filename).parent.parent / '05_query_pts' / Path(filename).name).with_suffix('.ply.npy')
        np.save(save_path, queries)

        complexes.update({Path(filename).stem: cell_complex})

    # evaluate query points
    evaluate()

    # reconstruct from cells
    for name in complexes:
        # find predictions in helsinki_noise_free\eval\eval\{name}
        npy_path = (Path('results') / 'p2s_max_model_249' / 'helsinki_noise_free/eval/eval/' / name).with_suffix(
            '.xyz.npy')
        npy_data = np.load(npy_path)
        indices_interior = np.where(npy_data > 0)[0]
        save_path = (npy_path.parent.parent / 'ply_reconstructed' / name).with_suffix('.ply')
        complexes[name].save_ply(save_path, indices_cells=indices_interior)


if __name__ == '__main__':
    reconstruct(data_paths='datasets/helsinki_noise_free/06_vertex_group/*.vg')
