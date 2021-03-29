from p2s.utils import points_to_surf_train

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
        '--indir', './datasets/helsinki_noise_0.001-0.005_no_bottom',
        '--refine', './p2s/models/p2s_max_model_249.pth',
        '--outdir', './p2s/models',
        '--logdir', './p2s/logs',
        '--trainset', 'trainset.txt',
        '--testset', 'valset.txt',
        '--nepoch', str(300),
        '--lr', str(0.01),  # relative to the checkpoint being refined
        '--scheduler_steps', str(100), str(200),  # relative to the checkpoint being refined
        '--debug', str(0),
        '--workers', str(7),  # 7 for normal machine; 22 for strong machine
        '--batchSize', str(101),  # 101 for normal machine; 1001 for strong machine
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
        '--outputs', 'imp_surf_magnitude', 'imp_surf_sign', 'patch_pts_ids', 'p_index',
    ]

    # train model on GT data with multiple query points per patch
    train_opt = points_to_surf_train.parse_arguments(train_params)
    points_to_surf_train.points_to_surf_train(train_opt)

    print('MeshNet training is finished!')
