import os
from p2s.base import evaluation


def evaluate_hausdorff_dist(data_dir, pred_dir, gt_dir, num_workers=6):
    csv_file = os.path.join(pred_dir, 'hausdorff_dist_pred_rec.csv')
    evaluation.mesh_comparison(
        new_meshes_dir_abs=pred_dir,
        ref_meshes_dir_abs=gt_dir,
        num_processes=num_workers,
        report_name=csv_file,
        samples_per_model=10000,  # randomness introduced
        dataset_file_abs=data_dir)


if __name__ == '__main__':
    dataset_name = 'helsinki_noise_free'  # famous_dense or helsinki_noise_free
    evaluate_hausdorff_dist(data_dir='datasets/{}/testset.txt'.format(dataset_name),
                            pred_dir='results/p2s_max_model_249/{}/eval/reconstructed'.format(dataset_name),
                            gt_dir='datasets/{}/03_meshes'.format(dataset_name))
