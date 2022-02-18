"""
Evaluate reconstruction results by Hausdorff distance.
"""

import os
from points2surf.source.base import evaluation


def evaluate_hausdorff_dist(data_dir, pred_dir, gt_dir, num_workers=6):
    """
    Evaluate Hausdorff distance between predicted and GT models.

    Parameters
    ----------
    data_dir: str
        Dir of input data
    pred_dir: str
        Dir of prediction results
    gt_dir: str
        Dir of ground truth
    num_workers: int
        Number of workers
    """
    csv_file = os.path.join(pred_dir, 'hausdorff_dist_pred_rec.csv')
    evaluation.mesh_comparison(
        new_meshes_dir_abs=pred_dir,
        ref_meshes_dir_abs=gt_dir,
        num_processes=num_workers,
        report_name=csv_file,
        samples_per_model=10000,
        dataset_file_abs=data_dir)


if __name__ == '__main__':
    dataset_name = 'helsinki_noise_free'
    model_name = 'p2s_max_model_249'
    evaluate_hausdorff_dist(data_dir=f'datasets/{dataset_name}/testset.txt',
                            pred_dir=f'results/{model_name}/{dataset_name}/eval/reconstructed',
                            gt_dir=f'datasets/{dataset_name}/03_meshes')
