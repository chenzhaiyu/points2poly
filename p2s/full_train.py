from p2s.utils import points_to_surf_train
import os

# When you see this error:
# 'Expected more than 1 value per channel when training...' which is raised by the BatchNorm1d layer
# for multi-gpu, use a batch size that can't be divided by the number of GPUs
# for single-gpu, use a straight batch size
# see https://github.com/pytorch/pytorch/issues/2584
# see https://forums.fast.ai/t/understanding-code-error-expected-more-than-1-value-per-channel-when-training/9257/12


if __name__ == '__main__':

    # settings for training p2s_max model
    train_params = [
        '--name', 'debug',
        '--indir', '../datasets/abc_train',
        '--testset', 'valset.txt',
        '--outdir', 'models',
        '--nepoch', str(10),
        '--lr', str(0.01),
        '--scheduler_steps', str(100), str(200),
        '--debug', str(0),
        '--workers', str(7),
        '--batchSize', str(101),
        '--points_per_patch', str(300),
        '--patches_per_shape', str(1000),
        '--sub_sample_size', str(1000),
        '--cache_capacity', str(30),
        '--patch_radius', str(0.0),
        '--single_transformer', str(0),
        '--shared_transformer', str(0),
        '--uniform_subsample', str(1),
        '--use_point_stn', str(0),
        '--net_size', str(1024),
        '--patch_center', 'mean',
        '--training_order', 'random_shape_consecutive',
        '--outputs', 'imp_surf_magnitude', 'imp_surf_sign', 'patch_pts_ids', 'p_index'
        ]

    # train model on GT data with multiple query points per patch
    train_opt = points_to_surf_train.parse_arguments(train_params)
    points_to_surf_train.points_to_surf_train(train_opt)
    
    print('MeshNet training is finished!')
