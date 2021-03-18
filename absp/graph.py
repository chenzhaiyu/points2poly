from pathlib import Path
import networkx as nx
import numpy as np
from sage.all import RR
from scipy.spatial import ConvexHull

from absp import attach_to_log

logger = attach_to_log()


class AdjacencyGraph:
    def __init__(self, graph=None):
        self.graph = graph
        self.uid = list(graph.nodes) if graph else None  # passed graph.nodes are sorted
        self.reachable = None  # for outer surface extraction
        self.non_reachable = None
        self._cached_interfaces = {}

    def load_graph(self, filepath):
        """
        Load graph from an external file.
        """
        filepath = Path(filepath)
        if filepath.suffix == '.adjlist':
            logger.info('loading graph from {}'.format(filepath))
            self.graph = nx.read_adjlist(filepath)
            self.uid = self._sort_uid()  # loaded graph.nodes are unordered string
        else:
            raise NotImplementedError('file format not supported: {}'.format(filepath.suffix))

    def assign_weights_to_n_links(self, cells, attribute='area_overlap', normalise=True, factor=1.0, engine='Qhull',
                                  cache_interfaces=False):
        """
        Assign weights to edges between every cell. weights is a dict with respect to each pair of nodes.
        """

        radius = [None] * len(self.graph.edges)
        area = [None] * len(self.graph.edges)
        volume = [None] * len(self.graph.edges)
        num_vertices = [None] * len(self.graph.edges)

        if attribute == 'radius_overlap':
            for i, (m, n) in enumerate(self.graph.edges):
                # compute interface
                interface = cells[self._uid_to_index(m)].intersection(cells[self._uid_to_index(n)])
                if cache_interfaces:
                    self._cached_interfaces[m, n] = interface  # uid pair as key
                radius[i] = RR(interface.radius())

            for i, (m, n) in enumerate(self.graph.edges):
                max_radius = max(radius)
                # small (sum of) overlapping radius -> large capacity -> small cost -> cut here
                self.graph[m][n].update({'capacity': ((max_radius - radius[
                    i]) / max_radius if normalise else max_radius - radius[i]) * factor})

        elif attribute == 'area_overlap':
            for i, (m, n) in enumerate(self.graph.edges):
                # compute interface
                interface = cells[self._uid_to_index(m)].intersection(cells[self._uid_to_index(n)])
                if cache_interfaces:
                    self._cached_interfaces[m, n] = interface  # uid pair as key
                if engine == 'Qhull':
                    # 'volume' is the area of the convex hull when input points are 2-dimensional
                    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html
                    area[i] = ConvexHull(interface.affine_hull_projection().vertices_list()).volume
                else:
                    # slower computation
                    area[i] = RR(interface.affine_hull_projection().volume())

            for i, (m, n) in enumerate(self.graph.edges):
                max_area = max(area)
                # balloon term
                # small (sum of) overlapping area -> large capacity -> small cost -> cut here
                # todo: adaptive threshold
                self.graph[m][n].update(
                    {'capacity': ((max_area - area[i]) / max_area if normalise else max_area - area[i]) * factor})

        elif attribute == 'vertices_overlap':
            # number of vertices on overlapping areas
            for i, (m, n) in enumerate(self.graph.edges):
                # compute interface
                interface = cells[self._uid_to_index(m)].intersection(cells[self._uid_to_index(n)])
                if cache_interfaces:
                    self._cached_interfaces[m, n] = interface  # uid pair as key
                num_vertices[i] = interface.n_vertices()

            for i, (m, n) in enumerate(self.graph.edges):
                max_vertices = max(num_vertices)
                # few number of vertices -> large capacity -> small cost -> cut here
                self.graph[m][n].update({'capacity': ((max_vertices - num_vertices[
                    i]) / max_vertices if normalise else max_vertices - num_vertices[i]) * factor})

        elif attribute == 'area_misalign':
            # area_misalign makes little sense as observed from the results
            logger.warning('attribute "area_misalign" is deprecated')

            # area of the mis-aligned regions from both cells
            for i, (m, n) in enumerate(self.graph.edges):
                # compute interface
                interface = cells[self._uid_to_index(m)].intersection(cells[self._uid_to_index(n)])
                if cache_interfaces:
                    self._cached_interfaces[m, n] = interface  # uid pair as key

                for facet_m in cells[self._uid_to_index(m)].facets():
                    for facet_n in cells[self._uid_to_index(n)].facets():
                        if facet_m.ambient_Hrepresentation()[0].A() == -facet_n.ambient_Hrepresentation()[0].A() and \
                                facet_m.ambient_Hrepresentation()[0].b() == -facet_n.ambient_Hrepresentation()[0].b():
                            # two facets coplanar
                            # area of the misalignment
                            if engine == 'Qhull':
                                area[i] = ConvexHull(
                                    facet_m.as_polyhedron().affine_hull_projection().vertices_list()).volume + ConvexHull(
                                    facet_n.as_polyhedron().affine_hull_projection().vertices_list()).volume - 2 * ConvexHull(
                                    interface.affine_hull_projection().vertices_list()).volume
                            else:
                                area[i] = RR(
                                    facet_m.as_polyhedron().affine_hull_projection().volume() +
                                    facet_n.as_polyhedron().affine_hull_projection().volume() -
                                    2 * interface.affine_hull_projection().volume())

            for i, (m, n) in enumerate(self.graph.edges):
                max_area = max(area)
                self.graph[m][n].update(
                    {'capacity': (area[i] / max_area if normalise else area[i]) * factor})

        elif attribute == 'volume_difference':
            # encourage partition between relatively a big cell and a small cell
            for i, (m, n) in enumerate(self.graph.edges):
                if engine == 'Qhull':
                    volume[i] = abs(ConvexHull(cells[self._uid_to_index(m)].vertices_list()).volume - ConvexHull(
                        cells[self._uid_to_index(n)].vertices_list()).volume) / max(
                        ConvexHull(cells[self._uid_to_index(m)].vertices_list()).volume,
                        ConvexHull(cells[self._uid_to_index(n)].vertices_list()).volume)
                else:
                    volume[i] = RR(
                        abs(cells[self._uid_to_index(m)].volume() - cells[self._uid_to_index(n)].volume()) / max(
                            cells[self._uid_to_index(m)].volume(), cells[self._uid_to_index(n)].volume()))

            for i, (m, n) in enumerate(self.graph.edges):
                max_volume = max(volume)
                # large difference -> large capacity -> small cost -> cut here
                self.graph[m][n].update(
                    {'capacity': (volume[i] / max_volume if normalise else volume[i]) * factor})

    def assign_weights_to_st_links(self, weights):
        """
        Assign weights to edges between each cell and the s-t nodes. weights is a dict in respect to each node.
        the weights can be the occupancy probability or the signed distance of the cells.
        """
        for i in self.uid:
            self.graph.add_edge(i, 's', capacity=weights[i])
            self.graph.add_edge(i, 't', capacity=1 - weights[i])  # make sure

    def cut(self):
        """
        Perform cutting operation.
        """
        cut_value, partition = nx.algorithms.flow.minimum_cut(self.graph, 's', 't')
        reachable, non_reachable = partition
        reachable.remove('s')
        non_reachable.remove('t')
        self.reachable = reachable
        self.non_reachable = non_reachable

        logger.info('cut_value: {}'.format(cut_value))
        logger.info('number of extracted cells: {}'.format(len(reachable)))
        return cut_value, reachable

    @staticmethod
    def _sorted_vertex_indices(adjacency_matrix):
        pointer = 0
        sorted_ = [pointer]
        for _ in range(len(adjacency_matrix[0]) - 1):
            connected = np.where(adjacency_matrix[pointer])[0]  # two elements
            if connected[0] not in sorted_:
                pointer = connected[0]
                sorted_.append(connected[0])
            else:
                pointer = connected[1]
                sorted_.append(connected[1])
        return sorted_

    def save_surface_obj(self, filepath, cells=None, engine='rendering'):
        """
        Save the outer surface to an OBJ file, from interfaces between cells being cut.
        Optional engine can be 'rendering', 'sorting' or 'projection'.
        """
        if not self.reachable:
            logger.error('no reachable cells. aborting')
            exit(1)
        elif not self.non_reachable:
            logger.error('no unreachable cells. aborting')
            exit(1)

        if not self._cached_interfaces and not cells:
            logger.error('neither cached interfaces nor cells are available. aborting')
            exit(1)

        if engine not in {'rendering', 'sorting', 'projection'}:
            logger.error('engine can be "rendering", "sorting" or "projection"')
            exit(1)

        surface = None
        surface_str = ''
        num_vertices = 0

        for edge in self.graph.edges:
            # facet is where one cell being outside and the other one being inside
            if edge[0] in self.reachable and edge[1] in self.non_reachable:
                # retrieve interface and orient as on edge[0]
                if self._cached_interfaces:
                    interface = self._cached_interfaces[edge[0], edge[1]] if (edge[0],
                                                                              edge[1]) in self._cached_interfaces else \
                        self._cached_interfaces[edge[1], edge[0]]
                else:
                    interface = cells[self._uid_to_index(edge[0])].intersection(cells[self._uid_to_index(edge[1])])

            elif edge[1] in self.reachable and edge[0] in self.non_reachable:
                # retrieve interface and orient as on edge[1]
                if self._cached_interfaces:
                    interface = self._cached_interfaces[edge[1], edge[0]] if (edge[1],
                                                                              edge[0]) in self._cached_interfaces else \
                        self._cached_interfaces[edge[0], edge[1]]
                else:
                    interface = cells[self._uid_to_index(edge[1])].intersection(cells[self._uid_to_index(edge[0])])

            else:
                # where no cut is made
                continue

            if engine == 'rendering':
                surface += interface.render_solid()

            elif engine == 'sorting':
                for v in interface.vertices():
                    surface_str += 'v {} {} {}\n'.format(float(v[0]), float(v[1]), float(v[2]))
                vertex_indices = [i + num_vertices + 1 for i in
                                  self._sorted_vertex_indices(interface.adjacency_matrix())]
                surface_str += 'f ' + ' '.join([str(f) for f in vertex_indices]) + '\n'
                num_vertices += len(vertex_indices)

            elif engine == 'projection':
                projection = interface.projection()
                polygon = projection.polygons[0]
                for v in projection.coords:
                    surface_str += 'v {} {} {}\n'.format(float(v[0]), float(v[1]), float(v[2]))
                surface_str += 'f ' + ' '.join([str(f + num_vertices + 1) for f in polygon]) + '\n'
                num_vertices += len(polygon)
        
        if engine == 'rendering':
            surface_obj = surface.obj_repr(surface.default_render_params())

            for o in range(len(surface_obj)):
                surface_str += surface_obj[o][0] + '\n'
                surface_str += '\n'.join(surface_obj[o][2]) + '\n'
                surface_str += '\n'.join(surface_obj[o][3]) + '\n'  # contents[o][4] are the interior facets

        # create the dir if not exists
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            f.writelines(surface_str)

    def draw(self):
        """
        Naively draw the graph with nodes represented by their UID.
        """
        import matplotlib.pyplot as plt
        plt.subplot(121)
        nx.draw(self.graph, with_labels=True, font_weight='bold')
        plt.show()

    def _uid_to_index(self, query):
        """
        Convert index in the node list to that in the cell list.
        The rationale behind is #nodes == #cells (when a primitive is settled down).
        :param query: query index in the node list.
        """
        return self.uid.index(query)

    def _index_to_uid(self, query):
        """
        Convert index to node UID.
        """
        return self.uid[query]

    def _sort_uid(self):
        """
        Sort UID for graph structure loaded from an external file.
        """
        return sorted([int(i) for i in self.graph.nodes])

    def to_indices(self, uids):
        """
        Convert UIDs to indices.
        """
        return [self._uid_to_index(i) for i in uids]

    def to_dict(self, weights_list):
        """
        Convert a weight list to weight dict keyed by self.uid.
        """
        return {self.uid[i]: weight for i, weight in enumerate(weights_list)}
