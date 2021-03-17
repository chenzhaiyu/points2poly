"""
Wrapper for p2s.
"""

import os
from source import points_to_surf_eval


def predict(dataset_name, opt=None):
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
