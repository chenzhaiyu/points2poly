"""
Cell complex from planar primitive arrangement.

Prerequisites:
* Planar primitives are extracted.

"""

from pathlib import Path
from random import random as random_  # todo: assign mtl to obj

import numpy as np
from tqdm import tqdm
import networkx as nx
from sage.all import polytopes, QQ, Polyhedron

from .logger import attach_to_log

logger = attach_to_log()


class CellComplex:

    def __init__(self, planes, bounds, initial_bound=None, build_graph=False):
        """
        :param planes: plana parameters. N * 4 array.
        :param bounds: corresponding bounding box bounds of the planar primitives. N * 2 * 3 array.
        :param initial_bound: optional. initial bound to partition. 2 * 3 array or None.
        :param build_graph: optional. build the cell adjacency graph if set True.
        """
        self.bounds = bounds  # numpy.array over RDF
        self.planes = planes  # numpy.array over RDF

        self.initial_bound = initial_bound if initial_bound else self._pad_bound(
            [np.amin(bounds[:, 0, :], axis=0), np.amax(bounds[:, 1, :], axis=0)],
            padding=0.20)
        self.cells = [self._construct_initial_cell()]  # list of QQ
        self.cells_bounds = [self.cells[0].bounding_box()]  # list of QQ

        if build_graph:
            self.graph = nx.Graph()
            self.graph.add_node(0)  # the initial cell
            self.index_node = 0  # unique for every cell ever generated
        else:
            self.graph = None

        self.constructed = False

    def _construct_initial_cell(self):
        """
        :return: Polyhedron object of the initial cell. a cuboid with 12 triangular facets.
        """
        return polytopes.cube(
            intervals=[[QQ(self.initial_bound[0][i]), QQ(self.initial_bound[1][i])] for i in range(3)])

    def prioritise_planes(self):
        """
        First, vertical planar primitives are accorded higher priority than horizontal or oblique ones
        to avoid incomplete partitioning due to missing data about building facades.
        Second, in the same priority class, planar primitives with larger areas are assigned higher priority
        than smaller ones, to make the final cell complex as compact as possible.
        Note that this priority setting is designed exclusively for building models.
        """
        logger.info('prioritising planar primitives')
        # compute the priority
        indices_vertical_planes = self._vertical_planes(slope_threshold=0.9)
        indices_sorted_planes = self._sort_planes()

        bool_vertical_planes = np.in1d(indices_sorted_planes, indices_vertical_planes)
        indices_priority = np.append(indices_sorted_planes[bool_vertical_planes],
                                     indices_sorted_planes[np.invert(bool_vertical_planes)])

        # reorder both the planes and their bounds
        self.planes = self.planes[indices_priority]
        self.bounds = self.bounds[indices_priority]

        logger.debug('ordered planes: {}'.format(self.planes))
        logger.debug('ordered bounds: {}'.format(self.bounds))

    def _vertical_planes(self, slope_threshold=0.9, epsilon=10e-5):
        """
        :return: the indices of the vertical planar primitives.
        """
        slope_squared = (self.planes[:, 0] ** 2 + self.planes[:, 1] ** 2) / (self.planes[:, 2] ** 2 + epsilon)
        return np.where(slope_squared > slope_threshold ** 2)[0]

    def _sort_planes(self):
        """
        :return: the indices by which the planar primitives are sorted based on their bounding box volume.
        """
        volume = np.prod(self.bounds[:, 1, :] - self.bounds[:, 0, :], axis=1)
        return np.argsort(volume)

    @staticmethod
    def _pad_bound(bound, padding=0.05):
        """
        :param bound: bound of the query planar primitive. 2 * 3 array.
        :param padding: optional. padding factor. float. defaults to 0.05.
        :return: padded bound.
        """
        extent = bound[1] - bound[0]
        return [bound[0] - extent * padding, bound[1] + extent * padding]

    def _bbox_intersect(self, bound):
        """
        :param bound: bound of the query planar primitive. 2 * 3 array.
        :return: indices of existing cells whose bounds intersect with that of the query primitive.
        """
        cells_bounds = np.array(self.cells_bounds)  # easier array manipulation
        bound = self._pad_bound(bound, padding=0.05)

        # intersection with existing cell AABB
        center_query = np.mean(bound, axis=0)  # 3,
        center_targets = np.mean(cells_bounds, axis=1)  # N * 3
        center_distance = np.abs(center_query - center_targets)  # N * 3

        extent_query = bound[1] - bound[0]  # 3,
        extent_targets = cells_bounds[:, 1, :] - cells_bounds[:, 0, :]  # N * 3

        # abs(center_distance) * 2 < (query extent + target extent) for every dimension -> intersection
        return np.where(np.all(center_distance * 2 < extent_query + extent_targets, axis=1))[0]

    @staticmethod
    def _inequalities(plane):
        """
        :param plane parameters. 4,.
        :return: inequalities defining two half-spaces separated by the plane
        """
        positive = [QQ(plane[-1]), QQ(plane[0]), QQ(plane[1]), QQ(plane[2])]
        negative = [QQ(-element) for element in positive]
        return positive, negative

    def _index_node_to_cell(self, query):
        """
        Convert index in the node list to that in the cell list.
        The rationale behind is #nodes == #cells (when a primitive is settled down).
        :param query: query index in the node list.
        """
        return list(self.graph.nodes).index(query)

    def construct(self):
        """
        Two-stage primitive-in-cell predicate. First, bounding boxes of primitive and existing cells are evaluated
        for possible intersection. Next, a strict intersection test is performed.

        Start partitioning. Generated cells are stored in self.cells.
        * query the bounding box intersection.
        * optional: intersection test for polygon and edge in each potential cell.
        * partition the potential cell into two. rewind if partition fails.
        """
        logger.info('constructing cell complex')

        for i, bound in enumerate(tqdm(self.bounds)):  # kinetic for each primitive
            # bounding box intersection test
            indices_cells = self._bbox_intersect(bound)  # indices of existing cells with potential intersections
            assert len(indices_cells), 'intersection failed! check the specified initial bound'

            # half-spaces defined by inequalities
            # no change_ring() here (instead, QQ() in _inequalities) speeds up 10x
            # init before the loop could possibly speed up a bit
            hspace_positive, hspace_negative = [Polyhedron(ieqs=[inequality]) for inequality in
                                                self._inequalities(self.planes[i])]

            # partition the intersected cells and their bounds while doing mesh slice plane
            indices_parents = []

            for index_cell in indices_cells:
                cell_positive = hspace_positive.intersection(self.cells[index_cell])
                cell_negative = hspace_negative.intersection(self.cells[index_cell])

                if cell_positive.dim() != 3 or cell_negative.dim() != 3:
                    # if cell_positive.is_empty() or cell_negative.is_empty():
                    """
                    cannot use is_empty() predicate for degenerate cases:
                        sage: Polyhedron(vertices=[[0, 1, 2]])
                        A 0-dimensional polyhedron in ZZ^3 defined as the convex hull of 1 vertex
                        sage: Polyhedron(vertices=[[0, 1, 2]]).is_empty()
                        False
                    """
                    continue

                # incrementally build the adjacency graph
                if self.graph is not None:
                    # append the two nodes (UID) being partitioned
                    self.graph.add_node(self.index_node + 1)
                    self.graph.add_node(self.index_node + 2)

                    # append the edge in between
                    self.graph.add_edge(self.index_node + 1, self.index_node + 2)

                    # get neighbours of the current cell from the graph
                    neighbours = self.graph[list(self.graph.nodes)[index_cell]]  # index in the node list

                    if neighbours:
                        # get the neighbouring cells to the parent
                        cells_neighbours = [self.cells[self._index_node_to_cell(n)] for n in neighbours]

                        # adjacency test between both created cells and their neighbours
                        # todo:
                        #   avoid 3d-3d intersection if possible. those unsliced neighbours connect with only one child
                        #   - reduce computation by half - can be further reduced using vertices/faces instead of
                        #   polyhedron intersection. those sliced neighbors connect with both children

                        for n, cell in enumerate(cells_neighbours):
                            if cell_positive.intersection(cell).dim() == 2:  # strictly a face
                                self.graph.add_edge(self.index_node + 1, list(neighbours)[n])
                            if cell_negative.intersection(cell).dim() == 2:
                                self.graph.add_edge(self.index_node + 2, list(neighbours)[n])

                    # update cell id
                    self.index_node += 2

                self.cells.append(cell_positive)
                self.cells.append(cell_negative)

                # incrementally cache the bounds for created cells
                self.cells_bounds.append(cell_positive.bounding_box())
                self.cells_bounds.append(cell_negative.bounding_box())

                indices_parents.append(index_cell)

            # delete the parent cells and their bounds. this does not affect the appended ones
            for index_parent in sorted(indices_parents, reverse=True):
                del self.cells[index_parent]
                del self.cells_bounds[index_parent]

                # remove the parent node (and its incident edges) in the graph
                if self.graph is not None:
                    self.graph.remove_node(list(self.graph.nodes)[index_parent])

        self.constructed = True
        logger.info('cell complex constructed')

    def visualise(self):
        if self.constructed:
            raise NotImplementedError
        else:
            raise RuntimeError('cell complex has not been constructed')

    @property
    def num_cells(self):
        # number of cells in the complex
        return len(self.cells)

    @property
    def num_plane(self):
        # excluding the initial bounding box
        return len(self.planes)

    def cell_representatives(self, location='center'):
        """
        :param location: 'center' represents the average of the vertices of the polyhedron;
        'centroid' represents the center of mass/volume.
        """
        if location == 'center':
            return [cell.center() for cell in self.cells]
        elif location == 'centroid':
            return [cell.centroid() for cell in self.cells]
        else:
            raise ValueError("expected 'mass' or 'centroid' as mode, got {}".format(location))

    def print_info(self):
        logger.info('number of planes: {}'.format(self.num_plane))
        logger.info('number of cells: {}'.format(self.num_cells))

    def save_npy(self, filepath):
        """
        Save the cell complex to an npy file.
        :param filepath: filepath.
        """
        if self.constructed:
            # create the dir if not exists
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            np.save(filepath, self.cells, allow_pickle=True)
        else:
            raise RuntimeError('cell complex has not been constructed')

    def save_obj(self, filepath, indices_cells=None, use_mtl=False):
        """
        Save polygon soup of indexed convexes to an obj file.
        :param filepath: filepath.
        :param indices_cells: indices of cells to save.
        :param use_mtl: write material info if set True.
        """
        if self.constructed:
            # create the dir if not exists
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            scene = None

            cells = [self.cells[i] for i in indices_cells] if indices_cells is not None else self.cells

            for cell in cells:
                scene += cell.render_solid()

            with open(filepath, 'w') as f:
                # directly save the obj string from scene.obj() will bring the inverted facets
                scene_obj = scene.obj_repr(scene.default_render_params())
                scene_str = ''
                for o in range(len(cells)):
                    scene_str += scene_obj[o][0] + '\n'
                    if use_mtl:
                        scene_str + scene_obj[o][1] + '\n'
                    scene_str += '\n'.join(scene_obj[o][2]) + '\n'
                    scene_str += '\n'.join(scene_obj[o][3]) + '\n'  # contents[o][4] are the interior facets
                f.writelines(scene_str)
        else:
            raise RuntimeError('cell complex has not been constructed')

    def save_plm(self, filepath, indices_cells=None):
        """
        Save polygon soup of indexed convexes to a plm file (polyhedron mesh in Mapple).
        :param filepath: filepath.
        :param indices_cells: indices of cells to save.
        """
        if self.constructed:
            num_vertices = 0
            info_vertices = ''
            info_facets = ''
            info_header = ''

            cells = [self.cells[i] for i in indices_cells] if indices_cells is not None else self.cells

            scene = None
            for cell in cells:
                scene += cell.render_solid()
                num_vertices += cell.n_vertices()

            info_header += '#vertices {}\n'.format(num_vertices)
            info_header += '#cells {}\n'.format(len(cells))

            with open(filepath, 'w') as f:
                contents = scene.obj_repr(scene.default_render_params())
                for o in range(len(cells)):
                    info_vertices += '\n'.join([st[2:] for st in contents[o][2]]) + '\n'
                    info_facets += str(len(contents[o][3])) + '\n'
                    for st in contents[o][3]:
                        info_facets += '\n'.join(str(len(st[2:].split()))) + ' '  # number of vertices on this facet
                        info_facets += ' '.join([str(int(n) - 1) for n in st[2:].split()]) + '\n'
                f.writelines(info_header + info_vertices + info_facets)

        else:
            raise RuntimeError('cell complex has not been constructed')
