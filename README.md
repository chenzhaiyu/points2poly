# Points2Poly

-----------

## Introduction

***Points2Poly*** is an implementation of the paper [*Reconstructing Compact Building Models from Point Clouds Using Deep Implicit Fields*](https://www.sciencedirect.com/science/article/pii/S0924271622002611), which incorporates learnable implicit surface representation into explicitly constructed geometry.

<p align="center">
<img src="https://raw.githubusercontent.com/chenzhaiyu/points2poly/master/docs/images/teaser.png" width="680"/>
</p>

Due to clutter concerns, **the core module is separately maintained in the [*abspy*](https://github.com/chenzhaiyu/abspy) repository** (also available as a [PyPI package](https://pypi.org/project/abspy/)), while this repository acts as a wrapper with additional sources and instructions in particular for building reconstruction.

## Prerequisites

The prerequisites are two-fold: one from `abspy` with functionalities on vertex group, cell complex, and adjacency graph; the other one from `points2surf` that facilitates occupancy estimation.

Clone this repository with submodules:
```bash
git clone --recurse-submodules https://github.com/chenzhaiyu/points2poly
```

In case you already cloned the repository but forgot `--recurse-submodules`:
```bash
git submodule update --init
```

### Requirements from `abspy` 

Follow the [instruction](https://github.com/chenzhaiyu/abspy#installation) to install `abspy` with its dependencies, while `abspy` itself can be easily installed via [PyPI](https://pypi.org/project/abspy/):
```bash
pip install abspy
```

###  Requirements from `points2surf`

Install the dependencies for `points2surf`:

```bash
pip install -r points2surf/requirements.txt
```

For training, make sure CUDA is available and enabled.
Navigate to [`points2surf/README.md`](https://github.com/ErlerPhilipp/points2surf) for more details on its requirements.

In addition, install dependencies for logging:

```bash
pip install -r requirements.txt
```

## Getting started

### Reconstrction demo 

Download a mini dataset of 6 buildings from the [Helsinki 3D city models](https://kartta.hel.fi/3d/), and a pre-trained full-view model:

```bash
python download.py dataset_name='helsinki_mini' model_name='helsinki_fullview'
```

Run reconstruction on the mini dataset:
```bash
python reconstruct.py dataset_name='helsinki_mini' model_name='helsinki_fullview'
```

Evaluate the reconstruction results by Hausdorff distance:

```bash
python evaluate.py dataset_name='helsinki_mini'
```

The reconstructed building models and statistics can be found under `./outputs/helsinki_mini/reconstructed`.

### Custom dataset

#### Reconstruction from custom point clouds

* **Convert point clouds into NumPy binary files** (`.npy`). Place point cloud files (e.g., `.ply`, `.obj`, `.stl` and `.off`) under `./datasets/{dataset_name}/00_base_pc` then run [`points2surf/make_pc_dataset.py`](https://github.com/ErlerPhilipp/points2surf/blob/master/make_pc_dataset.py), or manually do the conversion.

* **Extract planar primitives from point clouds with** [Mapple](https://github.com/LiangliangNan/Easy3D/releases/tag/v2.5.2). In Mapple, use `Point Cloud` - `RANSAC primitive extraction` to extract planar primitives, then save the vertex group files (`.vg` or `.bvg`) into `./datasets/{dataset_name}/06_vertex_group`.

* **Run reconstruction the same way as that in the demo**. Notice that, however, you might need to retrain a model that conforms to your data's characteristics.

#### Make training data

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

