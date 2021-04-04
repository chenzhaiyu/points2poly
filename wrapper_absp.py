"""
Wrapper for absp.

"""
import numpy as np

from utils import sigmoid
from absp import VertexGroup, CellComplex, AdjacencyGraph


def create_cell_complex(filepath_read_vg, filepath_write_candidate=None, prioritise_verticals=True, normalise_vg=False,
                        append_bottom=False):
    # load planes and bounds from vg data of a (complete) point cloud
    vertex_group = VertexGroup(filepath_read_vg)

    # normalise only if the dataset is created from point clouds instead of meshes
    if normalise_vg:
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

    # save candidate cells
    if filepath_write_candidate:
        filepath_write_candidate.parent.mkdir(parents=True, exist_ok=True)
        if filepath_write_candidate.suffix == '.plm':
            cell_complex.save_plm(filepath_write_candidate)
        elif filepath_write_candidate.suffix == '.npy':
            cell_complex.save_npy(filepath_write_candidate)
    cell_complex.print_info()
    return cell_complex


def create_query_points(cell_complex, filepath_write_query):
    queries = np.array(cell_complex.cell_representatives(location='center'), dtype=np.float32)

    # save the query points to 05_query_pts
    filepath_write_query.parent.mkdir(parents=True, exist_ok=True)
    np.save(filepath_write_query, queries)


def extract_surface(filepath_write_surface, cell_complex, sdf_values, graph_cut=False, coefficient=0.0010):
    if not graph_cut:
        # naive classification from SDF values
        indices_interior = np.where(sdf_values > 0)[0]
        cell_complex.save_plm(filepath_write_surface, indices_cells=indices_interior)

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
        _, reachable = adjacency_graph.cut()

        # write surface into obj file
        adjacency_graph.save_surface_obj(filepath=filepath_write_surface, cells=cell_complex.cells, engine='projection')
