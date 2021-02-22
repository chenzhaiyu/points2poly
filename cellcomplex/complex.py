"""
Cell complex from planar primitive arrangement.

Prerequisites:
* Planar primitives are extracted.

"""

import numpy as np
import trimesh
from .logger import get_logger
from tqdm import tqdm
from pathlib import Path

logger = get_logger()


class CellComplex:

    @trimesh.constants.log_time
    def __init__(self, planes, bounds, initial_bound=None, cell_bbox_type='AABB'):
        """
        :param planes: plana parameters. N * 4 array.
        :param bounds: corresponding bounding box bounds of the planar primitives. N * 2 * 3 array.
        :param initial_bound: optional. initial bound to partition. 2 * 3 array or None.
        :param cell_bbox_type: optional. cell bounding box type. 'AABB' or 'OBB' are supported. defaults to 'AABB'.
        """
        self.bounds = bounds
        self.planes = planes

        self.initial_bound = initial_bound if initial_bound else self._pad_bound(
            [np.amin(bounds[:, 0, :], axis=0), np.amax(bounds[:, 1, :], axis=0)],
            padding=0.20)  # sometimes ValueError('Input mesh must be watertight to cap slice'). try 0 or maybe 0.2
        self.cells = [self._construct_initial_cell()]

        self.cell_bbox_type = cell_bbox_type
        self.cells_bounds = [self._mesh_bound(self.cells[0])]

        self.constructed = False

    def _mesh_bound(self, mesh):
        """
        :param mesh: trimesh object.
        :return: mesh bounding box. 2 * 3 array.
        """
        # 'AABB' is faster to compute while 'OBB' reduce more false intersection
        if self.cell_bbox_type == 'AABB':
            return mesh.bounds
        elif self.cell_bbox_type == 'OBB':
            raise NotImplementedError('OBB bounding box type is not supported yet')
            # return mesh.bounding_box_oriented  # todo: catch OBB return
        else:
            raise TypeError(
                'cell_bbox_type should be either "AABB" or "OBB". got {} instead'.format(self.cell_bbox_type))

    def _construct_initial_cell(self):
        """
        :return: trimesh object of the initial cell. a cuboid with 12 triangular facets.
        """
        vertices = trimesh.bounds.corners(self.initial_bound)
        return trimesh.Trimesh(vertices=vertices,
                               faces=[[1, 2, 6, 5], [2, 3, 7, 6], [4, 7, 3, 0], [5, 4, 0, 1], [4, 5, 6, 7],
                                      [1, 0, 3, 2]], process=True)

    @trimesh.constants.log_time
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

        # reorder both the planes and the bounds
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

        if self.cell_bbox_type == 'AABB':
            # intersection with existing cell AABB
            center_query = np.mean(bound, axis=0)  # 3,
            center_targets = np.mean(cells_bounds, axis=1)  # N * 3
            center_distance = np.abs(center_query - center_targets)  # N * 3

            extent_query = bound[1] - bound[0]  # 3,
            extent_targets = cells_bounds[:, 1, :] - cells_bounds[:, 0, :]  # N * 3

            # abs(center_distance) * 2 < (query extent + target extent) for every dimension -> intersection
            return np.where(np.all(center_distance * 2 < extent_query + extent_targets, axis=1))[0]

        elif self.cell_bbox_type == 'OBB':
            raise NotImplementedError('OBB intersection is not implemented')
            # todo: 'OBB' intersection
            # https://stackoverflow.com/questions/47866571/simple-oriented-bounding-box-obb-collision-detection-explaining

    @staticmethod
    def _normal_and_origin(plane):
        """
        :param plane parameters. 4,.
        :return: normal and the point of cutoff coordinates. normal: 3,. origin: 3,.
        """
        # inclusive especially for axis-aligned plane
        assert plane[0] or plane[1] or plane[2], 'plane normal invalid'
        axis = np.argmax(plane[:3])  # the axis that has the most prominent projected normal vector
        origin = np.array([0.0, 0.0, 0.0])
        origin[axis] = -plane[3] / plane[axis]  # cutoff distance on this axis
        return plane[:3], origin

    @trimesh.constants.log_time
    def construct(self, force_convex_hull=False):
        """
        Two-stage primitive-in-cell predicate. First, bounding boxes of primitive and existing cells are evaluated
        for possible intersection. Next, a strict intersection test is performed.
        :param force_convex_hull: force convex hull calculation fro every generated cell if set True.
        otherwise only when necessary (non-watertight). enabling this may resolve some degenerate cases
        but slows down the computation slightly.

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

            # convert plane coefficients to normal and origin
            normal, origin = self._normal_and_origin(self.planes[i])

            # partition the intersected cells and their bounds while doing mesh slice plane
            indices_parents = []

            for index in indices_cells:
                try:
                    cell_positive = self.cells[index].slice_plane(origin, normal, cap=True)
                    cell_negative = self.cells[index].slice_plane(origin, -normal, cap=True)

                except (AttributeError, IndexError) as e:
                    # AttributeError is because no intersection between cell & primitive
                    # IndexError is because the intersection interface is too skinny for floating point error
                    logger.debug('cell {} for primitive {} slip from intersection: {}'.format(index, i, e))
                    continue

                """
                Qhull generates convex hull for non-watertight cell without triangulation (option 'Qt')
                or scaling (option 'QbB'). by default, Qhull merges facets to handle precision errors.
                triangulation would introduce non-watertight mesh again. scaling may reduce precision errors 
                if coordinate values vary widely. with option 'Pp', Qhull does not print statistics about 
                precision problems, and it removes some of the warnings.
                """
                if force_convex_hull:
                    cell_positive = trimesh.convex.convex_hull(cell_positive, qhull_options='Pp')
                    cell_negative = trimesh.convex.convex_hull(cell_negative, qhull_options='Pp')
                else:
                    # make sure the cell is watertight for future slicing
                    if not cell_positive.is_watertight:
                        cell_positive = trimesh.convex.convex_hull(cell_positive, qhull_options='Pp')
                    if not cell_negative.is_watertight:
                        cell_negative = trimesh.convex.convex_hull(cell_negative, qhull_options='Pp')

                self.cells.append(cell_positive)
                self.cells.append(cell_negative)

                # incrementally cache the bounds for created cells
                self.cells_bounds.append(self._mesh_bound(cell_positive))
                self.cells_bounds.append(self._mesh_bound(cell_negative))

                indices_parents.append(index)

            # delete the parent cells and their bounds. this does not affect the appended ones
            for index_parent in sorted(indices_parents, reverse=True):
                del self.cells[index_parent]
                del self.cells_bounds[index_parent]

        self.constructed = True
        logger.info('cell complex constructed')

    def visualise(self, index=None):
        if self.constructed:
            # assign random color to each mesh
            for mesh in self.cells:
                mesh.visual.vertex_colors = trimesh.visual.random_color()
            scene = self.cells[index] if index else trimesh.Scene(self.cells)
            scene.show()
        else:
            raise RuntimeError('cell complex has not been constructed')

    @property
    def num_cells(self):
        return len(self.cells)

    @property
    def num_plane(self):
        # excluding the initial bounding box
        return len(self.planes)

    @property
    def bound(self):
        return self.initial_bound

    def cell_centers(self, mode='mass'):
        """
        :param mode: 'mass' represents the center of mass/volume; 'centroid' represents
        the average of the triangle centroids weighted by the area of each triangle.
        """
        if mode == 'mass':
            return [cell.center_mass for cell in self.cells]
        elif mode == 'centroid':
            return [cell.centroid for cell in self.cells]
        else:
            raise ValueError("expected 'mass' or 'centroid' as mode, got {}".format(mode))

    def cell_samples(self, num_samples):
        """
        :param num_samples: number of samples per cell.
        :return samples. m * n * 3 array.
        """
        return [trimesh.sample.volume_mesh(mesh, num_samples) for mesh in self.cells]

    def print_info(self):
        logger.info('number of planes: {}'.format(self.num_plane))
        logger.info('number of cells: {}'.format(self.num_cells))
        logger.debug('bound of the complex: {}'.format(self.bound))

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

    def save_ply(self, filepath, indices_cells=None):
        """
        Save polygon soup of indexed convexes to a ply file.
        :param filepath: filepath.
        :param indices_cells: indices of cells to save.
        """
        if self.constructed:
            # create the dir if not exists
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            mesh = trimesh.util.concatenate(
                [self.cells[i] for i in indices_cells]) if indices_cells is not None else trimesh.util.concatenate(
                self.cells)
            mesh.export(filepath, file_type='ply')

        else:
            raise RuntimeError('cell complex has not been constructed')

    def save_plm(self, filepath, indices_cells=None, polygonal=False):
        """
        Save polygon soup of indexed convexes to a plm file (polygonal mesh in Mapple).
        :param filepath: filepath.
        :param indices_cells: indices of cells to save.
        :param polygonal: merge coplanar triangular faces into polygonal ones if set True.
        """
        if self.constructed:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            num_vertices = 0
            shift = 0
            info_vertices = ''
            info_faces = ''
            info_header = ''

            cells = [self.cells[i] for i in indices_cells] if indices_cells is not None else self.cells
            for cell in cells:
                num_vertices += len(cell.vertices)
                for vertex in cell.vertices:
                    info_vertices += ' '.join(map(str, vertex)) + '\n'

                if not polygonal:
                    info_faces += '{}\n'.format(len(cell.faces))
                    for face in cell.faces:
                        info_faces += '{} '.format(len(face)) + ' '.join(map(str, face + shift)) + '\n'
                    shift += len(cell.vertices)

                else:
                    # grouping coplanar adjacent triangular faces to polygons
                    coplanar_triangles = []
                    if cell.facets:
                        coplanar_triangles = np.hstack(cell.facets)  # indices of faces that has coplanar adjacent faces
                    singleton = set(range(len(cell.faces))) - set(coplanar_triangles)  # indices of remaining triangles

                    # link the boundary edges to form polygons for current cell
                    facets_boundary = cell.facets_boundary  # edges which represent the boundary of each coplanar faces
                    polygons = [trimesh.graph.traversals(boundary, mode='dfs')[0] for boundary in facets_boundary]

                    # check normal orientation with the triangle edges and reverse the polygon of wrong orientation
                    # this is more robust than computing the normal with the first three vertices, which presumes
                    # the polygon is convex and may mis-render the degenerated case due to floating point errors
                    for i, facet in enumerate(cell.facets):
                        edges = np.vstack(
                            [[[face[0], face[1]], [face[1], face[2]], [face[2], face[0]]] for face in
                             cell.faces[facet]])
                        if polygons[i][:2].tolist() not in edges.tolist():
                            polygons[i] = polygons[i][::-1]

                    # append the triangles, whose orientation should be good
                    polygons += [np.array(cell.faces[i]) for i in singleton]

                    info_faces += '{}\n'.format(len(polygons))
                    for polygon in polygons:
                        info_faces += '{} '.format(len(polygon)) + ' '.join(map(str, polygon + shift)) + '\n'
                    shift += len(cell.vertices)

            info_header += '#vertices {}\n'.format(num_vertices)
            info_header += '#cells {}\n'.format(len(cells))

            with open(filepath, 'w') as f:
                f.writelines(info_header + info_vertices + info_faces)

        else:
            raise RuntimeError('cell complex has not been constructed')
