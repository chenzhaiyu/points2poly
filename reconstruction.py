"""
Piece-wise planar reconstruction.

"""
import os
import glob
import numpy as np
from pathlib import Path

from absp import VertexGroup, CellComplex, AdjacencyGraph
from absp import attach_to_log

from source import points_to_surf_eval
from source.base import evaluation
from source import sdf

logger = attach_to_log(filepath='reconstruction.log')

dataset_name = 'helsinki_noise_free'


def evaluate(opt=None):
    global dataset_name
    if not opt:
        class OPT:
            pass

        opt = OPT
        opt.seed = 40938661
        opt.indir = 'datasets'
        opt.outdir = 'results'
        opt.modeldir = 'models'
        opt.dataset = '{}/testset.txt'.format(dataset_name)
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
        print('evaluating on dataset {}'.format(dataset))
        opt.indir = os.path.join(indir_root, os.path.dirname(dataset))
        opt.outdir = os.path.join(outdir_root, os.path.dirname(dataset))
        opt.dataset = os.path.basename(dataset)

        # evaluate
        opt.reconstruction = False
        points_to_surf_eval.points_to_surf_eval(opt)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def reconstruct(data_paths):
    global dataset_name
    complexes = {}
    for filename in glob.glob(data_paths):
        # load planes and bounds from vg data of a (complete) point cloud
        vertex_group = VertexGroup(filename)
        planes, bounds, points = np.array(vertex_group.planes), np.array(vertex_group.bounds), np.array(vertex_group.points_grouped, dtype=object)

        # construct cell complex and extract the cell centers as query points
        cell_complex = CellComplex(planes, bounds, points, build_graph=True)
        cell_complex.refine_planes(epsilon=0.005)
        cell_complex.prioritise_planes()
        cell_complex.construct()
        # cell_complex.visualise()

        # save candidate cells
        save_candidates = False
        if save_candidates:
            cell_complex.save_plm(
                (Path('results') / 'p2s_max_model_249' / '{}/eval/candidates'.format(dataset_name) / Path(filename).name).with_suffix('.plm'))
        cell_complex.print_info()

        queries = np.array(cell_complex.cell_representatives(location='center'), dtype=np.float32)

        # save the query points to 05_query_pts
        save_path = (Path(filename).parent.parent / '05_query_pts' / Path(filename).name).with_suffix('.ply.npy')
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, queries)

        complexes.update({Path(filename).stem: cell_complex})

    # evaluate query points
    evaluate()

    # reconstruct from cells
    for name in complexes:
        # find predictions in helsinki_noise_free\eval\eval\{name}
        npy_path = (Path('results') / 'p2s_max_model_249' / '{}/eval/eval/'.format(dataset_name) / name).with_suffix(
            '.xyz.npy')
        npy_data = np.load(npy_path)

        naive_classification = False
        if naive_classification:
            indices_interior = np.where(npy_data > 0)[0]
            save_path = (npy_path.parent.parent / 'reconstructed_naive' / name).with_suffix('.plm')
            complexes[name].save_plm(save_path, indices_cells=indices_interior)

        graph_cut = True
        if graph_cut:
            adjacency_graph = AdjacencyGraph(complexes[name].graph)

            # compute the volumes
            engine = 'Qhull'
            volume_factor = 10e5
            if engine == 'Qhull':
                from scipy.spatial import ConvexHull
                volumes = [ConvexHull(cell.vertices_list()).volume * volume_factor for cell in complexes[name].cells]
            else:
                from sage.all import RR
                volumes = [RR(cell.volume()) * volume_factor for cell in complexes[name].cells]

            weights_dict = adjacency_graph.to_dict(sigmoid(npy_data * volumes))

            attribute = 'area_overlap'
            factor = 0.0009
            adjacency_graph.assign_weights_to_n_links(complexes[name].cells, attribute=attribute, factor=factor)  # 0.0001
            adjacency_graph.assign_weights_to_st_links(weights_dict)
            _, reachable = adjacency_graph.cut()

            save_path = (npy_path.parent.parent / 'reconstructed_gcut_{}_{}'.format(attribute, factor) / name).with_suffix('.obj')
            complexes[name].save_obj(save_path, indices_cells=adjacency_graph.to_indices(reachable))


if __name__ == '__main__':
    reconstruct(data_paths='datasets/{}/06_vertex_group/*.vg'.format(dataset_name))
