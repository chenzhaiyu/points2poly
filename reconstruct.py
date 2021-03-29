import glob
import numpy as np
from pathlib import Path
import pickle

from wrapper_absp import create_cell_complex, create_query_points, extract_surface
from wrapper_p2s import predict
from absp import attach_to_log

attach_to_log(filepath='reconstruction.log')


def reconstruct_full(dataset_paths, complexes_path=None):
    # full evaluation starting from point clouds

    # create cell complexes and save query points (.npy)
    complexes = {}
    for filepath in glob.glob(dataset_paths):
        filepath = Path(filepath)

        # prioritise_verticals for buildings
        # normalise vg only for dataset created from point clouds instead of meshes
        filepath_write_candidate = filepath.with_suffix('.plm')
        cell_complex = create_cell_complex(filepath, filepath_write_candidate=filepath_write_candidate, prioritise_verticals=True,
                                           normalise_vg=False)

        complexes.update({filepath.stem: cell_complex})
        create_query_points(cell_complex,
                            filepath_write_query=(filepath.parent.parent / '05_query_pts' / filepath.name).with_suffix(
                                '.ply.npy'))

    # dump complexes
    if complexes_path:
        Path(complexes_path).parent.mkdir(parents=True, exist_ok=True)
        with open(complexes_path, 'wb') as f_complexes:
            pickle.dump(complexes, f_complexes)

    # batch prediction and save sdf values (.npy)
    predict(dataset_name)

    # extract surfaces (.obj)
    for name in complexes:
        sdf_path = (Path('results') / 'p2s_max_model_249' / '{}/eval/eval/'.format(dataset_name) / name).with_suffix(
            '.xyz.npy')
        sdf_values = np.load(sdf_path)
        extract_surface((sdf_path.parent.parent / 'reconstructed' / name).with_suffix('.obj'), complexes[name],
                        sdf_values, graph_cut=True, coefficient=0.005)


def reconstruct_surface(complexes_path):
    # evaluation of surface extraction only. cell complexes and sdf predictions are loaded off-the-shelf.

    # load cell complexes
    with open(complexes_path, 'rb') as f_complexes:
        complexes = pickle.load(f_complexes)
        print('{} loaded'.format(complexes_path))

    for name in complexes:
        # load prediction results
        sdf_path = (Path('results') / 'p2s_max_model_249' / '{}/eval/eval/'.format(dataset_name) / name).with_suffix(
            '.xyz.npy')
        sdf_values = np.load(sdf_path)

        # surface extraction
        extract_surface((sdf_path.parent.parent / 'reconstructed' / name).with_suffix('.obj'), complexes[name],
                        sdf_values, graph_cut=True, coefficient=0.0010)


if __name__ == '__main__':
    dataset_name = 'helsinki_noise_free'  # famous_dense or helsinki_noise_free

    reconstruct_full(dataset_paths='datasets/{}/06_vertex_group/*.vg'.format(dataset_name),
                     complexes_path='results/p2s_max_model_249/{}/eval/complexes.dictionary'.format(dataset_name))

    # reconstruct_surface(
    #     complexes_path='results/p2s_max_model_249/{}/eval/complexes.dictionary'.format(dataset_name))
