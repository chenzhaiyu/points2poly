from p2s.utils import points_to_surf_train
import os

# When you see this error:
# 'Expected more than 1 value per channel when training...' which is raised by the BatchNorm1d layer
# for multi-gpu, use a batch size that can't be divided by the number of GPUs
# for single-gpu, use a straight batch size
# see https://github.com/pytorch/pytorch/issues/2584
# see https://forums.fast.ai/t/understanding-code-error-expected-more-than-1-value-per-channel-when-training/9257/12


def full_train(opt):

    points_to_surf_train.points_to_surf_train(opt)


if __name__ == '__main__':

    model_name = 'debug'
    dataset = 'helsinki_noise_0.001'
    base_dir = '../datasets'
    in_dir_train = os.path.join(base_dir, dataset)

    train_set = 'trainset.txt'
    val_set = 'valset.txt'
    test_set = 'testset.txt'

    # features = ['imp_surf', 'patch_pts_ids', 'p_index']  # l2-loss
    features = ['imp_surf_magnitude', 'imp_surf_sign', 'patch_pts_ids', 'p_index']  # l2-loss + BCE-loss

    # workers = 22  # for strong training machine
    workers = 7  # for typical PC

    # batch_size = 501  # ~7.5 GB memory on 4 2080 TI for 300 patch points + 1000 sub-sample points
    # batch_size = 3001  # ~10 GB memory on 4 2080 TI for 50 patch points + 200 sub-sample points
    batch_size = 50  # ~7 GB memory on 1 1070 for 300 patch points + 1000 sub-sample points

    # grid_resolution = 256  # quality like in the paper
    grid_resolution = 128  # quality for a short test
    rec_epsilon = 3
    certainty_threshold = 13
    sigma = 5

    fixed_radius = False
    patch_radius = 0.1 if fixed_radius else 0.0
    single_transformer = 0
    shared_transformer = 0

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    train_params = [
        '--name', model_name,
        '--desc', model_name,
        '--indir', in_dir_train,
        '--outdir', 'models',
        '--trainset', train_set,
        '--testset', val_set,
        '--net_size', str(1024),
        '--nepoch', str(10),
        '--lr', str(0.01),
        '--debug', str(0),
        '--workers', str(workers),
        '--batchSize', str(batch_size),
        '--points_per_patch', str(300),
        '--patches_per_shape', str(1000),
        '--sub_sample_size', str(1000),
        '--cache_capacity', str(10),
        '--patch_radius', str(patch_radius),
        '--single_transformer', str(single_transformer),
        '--shared_transformer', str(shared_transformer),
        '--patch_center', 'mean',
        '--training_order', 'random_shape_consecutive',
        '--use_point_stn', str(1),
        '--uniform_subsample', str(0),
        '--outputs',
    ]
    train_params += features

    # train model on GT data with multiple query points per patch
    train_opt = points_to_surf_train.parse_arguments(train_params)
    full_train(train_opt)
    
    print('MeshNet training is finished!')
