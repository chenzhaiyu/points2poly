# Points2Poly

-----------

## Introduction

***Points2Poly*** is an implementation of the paper [*Reconstructing Compact Building Models from Point Clouds Using Deep Implicit Fields*](https://www.sciencedirect.com/science/article/pii/S0924271622002611). This implementation incorporates learnable implicit surface representation into explicitly constructed geometry.

<p align="center">
<img src="https://raw.githubusercontent.com/chenzhaiyu/points2poly/master/docs/images/teaser.png" width="680"/>
</p>

Due to clutter concerns, **the core module is separately maintained in the [*abspy*](https://github.com/chenzhaiyu/abspy) repository** (also available as a [PyPI package](https://pypi.org/project/abspy/)), while this repository acts as a wrapper with additional sources and instructions in particular for building reconstruction.

## Prerequisites

The prerequisites consist of two parts: one from `abspy` that provides functionalities on vertex group, cell complex, and adjacency graph; the other one from `points2surf` that facilitates occupancy estimation.

Clone this repository with submodules:
```bash
git clone --recurse-submodules https://github.com/chenzhaiyu/points2poly
```

In case you already cloned the repository but forgot `--recurse-submodules`:
```bash
git submodule update --init
```

### Requirements from `abspy` 

Follow this [instruction](https://github.com/chenzhaiyu/abspy#installation) to install `abspy` with its appendencies, while `abspy` itself can be easily installed via PyPI:
```bash
pip install abspy
```

###  Requirements from `points2surf`

Install the requirements listed in `points2surf/requirements.txt` with PyPI:

```bash
pip install -r points2surf/requirements.txt
```

For training, make sure CUDA is available and enabled.
Navigate to `points2surf/README.md` for more details on its requirements.

## Getting started

### Reconstrction demo 

Download a mini dataset (point clouds, meshes, etc.) that consists of 6 buildings from the [Helsinki 3D city models](https://kartta.hel.fi/3d/), and a pre-trained full-view model:

```bash
python download.py dataset_name='helsinki_mini' model_name='helsinki_fullview'
```

Run reconstruction on the mini dataset:
```bash
python reconstruct.py dataset_name='helsinki_mini' model_name='helsinki_fullview'
```

Evaluate the reconstruction results by Hausdorff distance:

```bash
# change {dir_reconstructed} to 'outputs/.../reconstructed' where the reconstructed buildings are saved
python evaluate.py dataset_name='helsinki_mini', result_dir={dir_reconstructed}
```

The reconstructed buildings can be found under `{dir_reconstructed}`.

### Custom dataset

#### Reconstruction from your own point clouds

* Convert point clouds into NumPy binary format (`.npy`). Place your point cloud files (e.g., `.ply`, `.obj`, `.stl` and `.off`) under `./datasets/{dataset_name}/00_base_pc` then run [`points2surf/make_pc_dataset.py`](https://github.com/ErlerPhilipp/points2surf/blob/master/make_pc_dataset.py), or manually do the conversion.

* Extract planar primitives from point clouds with [Mapple](https://3d.bk.tudelft.nl/liangliang/software.html). You can either build Mapple as an application from [Easy3D](https://github.com/LiangliangNan/Easy3D), or directly download one of the [executables](https://github.com/LiangliangNan/Easy3D/releases/tag/v2.4.7). In Mapple, use `Point Cloud` - `RANSAC primitive extraction` to extract planar primitives, and then save the results (`.vg` or `.bvg`) into `./datasets/{dataset_name}/06_vertex_group`.

* Then you should be able to reconstruct your point clouds the same way you did with the demo data. Notice that, however, you might need to retrain a model that conforms to your data's characteristics.

#### Make your training data

Prepare meshes and place them under `datasets/{dataset_name}` that mimic the structure of the provided data. Refer to this [instruction](https://github.com/ErlerPhilipp/points2surf#make-your-own-datasets) for creating training data through [BlenSor](https://www.blensor.org/) simulation. 

## TODOs

- [x] Separate `abspy`/`points2surf` from `points2poly` wrappers
- [x] Config with hydra
- [x] Short tutorial on how to get started
- [ ] Host generated data

## License

[MIT](https://raw.githubusercontent.com/chenzhaiyu/points2poly/main/LICENSE)

## Acknowledgement
The implementation of *Points2Poly* has greatly benefited from [Points2Surf](https://github.com/ErlerPhilipp/points2surf). In addition, the implementation of the *abspy* submodule is backed by great open-source libraries inlcuding [SageMath](https://www.sagemath.org/), [NetworkX](https://networkx.org/), and [Easy3D](https://github.com/LiangliangNan/Easy3D).

## Citation

If you use *Points2Poly* in a scientific work, please consider citing the paper:

```bibtex
@article{chen2022points2poly,
  title = {Reconstructing compact building models from point clouds using deep implicit fields},
  journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
  volume = {194},
  pages = {58-73},
  year = {2022},
  issn = {0924-2716},
  doi = {https://doi.org/10.1016/j.isprsjprs.2022.09.017},
  url = {https://www.sciencedirect.com/science/article/pii/S0924271622002611},
  author = {Zhaiyu Chen and Hugo Ledoux and Seyran Khademi and Liangliang Nan}
}
```

