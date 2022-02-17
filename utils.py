"""
Wrapper utilities for
  * [abspy](https://github.com/chenzhaiyu/abspy) and
  * [points2surf](https://github.com/ErlerPhilipp/points2surf).

[points2surf] is wrapped to infer signed distance values given a cell complex and trained weights.
[abspy] is wrapped to create the cell complex, the further query points, and extract the surface.
"""

import numpy as np
from omegaconf import DictConfig, ListConfig

from abspy import VertexGroup, CellComplex, AdjacencyGraph
from points2surf.source import points_to_surf_eval


def sigmoid(x):
    """
    Sigmoid function.

    Parameters
    ----------
    x: float
        Scalar input

    Returns
    -------
    as_float: float
        Value of sigmoid
    """
    # can safely ignore RuntimeWarning: overflow encountered in exp
    return 1 / (1 + np.exp(-x))


def create_cell_complex(filepath_vg, filepath_out=None, prioritise_verticals=True,
                        normalise=False, append_bottom=False):
    """
    Create cell complex from vertex group file (.vg).

    Parameters
    ----------
    filepath_vg: Path
        Filepath to input vertex group file (.vg)
    filepath_out: Path
        Filepath to save cell complex,
        either as Easy3D polyhedra mesh (.plm) or as Numpy array (.npy)
    prioritise_verticals: bool
        Prioritise vertical primitives if set True
    normalise: bool
        Normalise the vertex group to centroid and unit scale if set True
    append_bottom: bool
        Append a bottom plane to close the model if set True

    Returns
    -------
    as_object: CellComplex object
        Cell complex
    """
    # load planes and bounds from vg data of a (complete) point cloud
    vertex_group = VertexGroup(filepath_vg)

    # normalise only if the dataset is created from point clouds instead of meshes
    if normalise:
        vertex_group.normalise_to_centroid_and_scale()

    planes, bounds, points = np.array(vertex_group.planes), np.array(vertex_group.bounds), np.array(
        vertex_group.points_grouped, dtype=object)

    # construct cell complex and extract the cell centers as query points
    if append_bottom:
        additional_planes = [[0, 0, 1, -bounds[:, 0, 2].min()],  # bottom
                             # [0, 0, 1, -bounds[:, 1, 2].max()],  # top
                             # [0, 0, 1, -bounds[:, 0, 0].min()],  # left
                             # [0, 0, 1, -bounds[:, 1, 0].max()],  # right
                             # [0, 0, 1, -bounds[:, 0, 1].min()],  # front
                             # [0, 0, 1, -bounds[:, 1, 1].max()],  # back
                             ]
    else:
        additional_planes = None

    cell_complex = CellComplex(planes, bounds, points, additional_planes=additional_planes, build_graph=True)
    cell_complex.refine_planes(epsilon=0.005)
    cell_complex.prioritise_planes(prioritise_verticals=prioritise_verticals)
    cell_complex.construct()

    if filepath_out:
        # save candidate cells to files
        filepath_out.parent.mkdir(parents=True, exist_ok=True)
        if filepath_out.suffix == '.plm':
            cell_complex.save_plm(filepath_out)
        elif filepath_out.suffix == '.npy':
            cell_complex.save_npy(filepath_out)
    cell_complex.print_info()
    return cell_complex


def create_query_points(cell_complex, filepath_query, location='center'):
    """
    Create query points from the representative of each cell in the complex.

    Parameters
    ----------
    cell_complex: CellComplex object
        Cell complex
    filepath_query: Path
        Filepath to write query points
    location: str
        Location of the representative point, can be 'center' or 'centroid'
    """
    queries = np.array(cell_complex.cell_representatives(location=location), dtype=np.float32)

    # save the query points to numpy file (e.g., under ./05_query_pts)
    filepath_query.parent.mkdir(parents=True, exist_ok=True)
    np.save(filepath_query, queries)


def extract_surface(filepath_surface, cell_complex, sdf_values, graph_cut=True, coefficient=0.0010):
    """
    Extract surface using graph cut.

    Parameters
    ----------
    filepath_surface: Path
        Filepath to save extracted surface
    cell_complex: CellComplex object
        Cell complex
    sdf_values: (n,) float
        Inferred signed distance values corresponding to candidates in the complex
    graph_cut: bool
        Use graph cut if set True (should always), use naive sign classification if set False (for debugging only)
    coefficient: float
        lambda coefficient for the complexity term
    """
    if not graph_cut:
        # naive classification from SDF values (for debugging only)
        indices_interior = np.where(sdf_values > 0)[0]
        cell_complex.save_plm(filepath_surface, indices_cells=indices_interior)

    else:
        # graph cut optimization
        adjacency_graph = AdjacencyGraph(cell_complex.graph)

        # fidelity term
        volumes = cell_complex.volumes(multiplier=10e5)
        weights_dict = adjacency_graph.to_dict(sigmoid(sdf_values * volumes))

        # graph partitioning
        attribute = 'area_overlap'
        adjacency_graph.assign_weights_to_n_links(cell_complex.cells, attribute=attribute, factor=coefficient,
                                                  cache_interfaces=True)
        adjacency_graph.assign_weights_to_st_links(weights_dict)
        _, _ = adjacency_graph.cut()

        # write surface into obj file
        adjacency_graph.save_surface_obj(filepath=filepath_surface, cells=cell_complex.cells,
                                         engine='projection')


def infer_sdf(cfg: DictConfig):
    """
    Infer SDF from point clouds.

    This assumes the points2surf network has been trained.
    The inferred SDF are represented by signed distance values of pre-sampled query points.

    Parameters
    ----------
    cfg: DictConfig
        Hydra configuration
    """
    print('evaluating on dataset {}'.format(cfg.dataset_name))
    points_to_surf_eval.points_to_surf_eval(cfg)
