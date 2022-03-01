# Points2Poly

-----------

## Introduction

***Points2Poly*** is the implementation of the compact building surface reconstruction method described in this [arxiv paper](https://arxiv.org/abs/2112.13142). This implementation incorporates learnable implicit surface representation into explicitly constructed geometry.

<p align="center">
<img src="https://raw.githubusercontent.com/chenzhaiyu/points2poly/master/docs/images/teaser.png" width="680"/>
</p>

Due to clutter concerns, **the core module is separately maintained in the [*abspy*](https://github.com/chenzhaiyu/abspy) repository** (also available as a [PyPI package](https://pypi.org/project/abspy/)), while this repository acts as a wrapper with additional sources and instructions in particular for building reconstruction. **The wrapper code is being cleaned.**

## Requirements

The requirements consist of two parts: one from `abspy` that provides functionalities on vertex group, cell complex, and adjacency graph; the other one from `points2surf` that facilitates occupancy estimation.

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

For training the neural network, make sure CUDA is available and enabled.
You can navigate to `points2surf/README.md` for more details on its requirements.

## Getting started

### Reconstrction demo with pre-trained models 

Download a mini dataset (point clouds, meshes, etc.) that consists of 6 buildings from the [Helsinki 3D city models](https://kartta.hel.fi/3d/), and a pre-trained full-view model:

```bash
python download.py dataset_name='helsinki_mini' model_name='helsinki_fullview'
```

Run reconstruction on the mini dataset:
```bash
python reconstruct.py dataset_name='helsinki_mini' model_name='helsinki_fullview'
```

Reconstructed buildings are saved to `outputs/{YYYY-MM-DD}/{HH-MM-SS}/{model_name-dataset_name}/reconstructed`.

### Custom dataset

Prepare (building) meshes and place them under `datasets/{dataset_name}` that mimic the structure of the provided data. Refer to this [instruction](https://github.com/ErlerPhilipp/points2surf/tree/2af6e0facf58422ed12e0c676c70199cd0dfbb43#make-your-own-datasets) for creating training data of compatible format through simulation. Train the `points2surf` neural network on the custom dataset for occupancy estimation. After training, run reconstruction with the trained model with `reconstruct.py`, and then evaluate the reconstruction results with `evaluate.py`.

## TODOs

- [x] Separate `abspy`/`points2surf` from `points2poly` wrappers
- [x] Config with hydra
- [x] Short tutorial on how to get started
- [ ] Host generated data

## License

[MIT](https://raw.githubusercontent.com/chenzhaiyu/points2poly/main/LICENSE)

## Acknowledgement
The implementation of *Points2Poly* has greatly benefited from [Points2Surf](https://github.com/ErlerPhilipp/points2surf). In addition, the implementation of the *abspy* submodule is backed by existing great open-source libraries, such as [SageMath](https://www.sagemath.org/), [NetworkX](https://networkx.org/), and [Easy3D](https://github.com/LiangliangNan/Easy3D).

## Citation

If you use *Points2Poly* in a scientific work, please consider citing it:

```bibtex
@article{chen2021reconstructing,
  title={Reconstructing Compact Building Models from Point Clouds Using Deep Implicit Fields},
  author={Chen, Zhaiyu and Khademi, Seyran and Ledoux, Hugo and Nan, Liangliang},
  journal={arXiv preprint arXiv:2112.13142},
  year={2021}
}
```
