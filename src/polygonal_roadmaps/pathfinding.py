from graph_tool.all import AStarVisitor, StopSearch, astar_search
import graph_tool
from dataclasses import dataclass
import numpy as np


class NavVisitor(AStarVisitor):
    def __init__(self, touched_v, touched_e, goal):
        self.touched_e = touched_e
        self.touched_v = touched_v
        self.target = goal

    def discover_vertex(self, u):
        self.touched_v[u] = True
    
    def examine_edge(self, e):
        self.touched_e[e] = True
    
    def edge_relaxed(self, e):
        if e.target() == self.target:
            raise StopSearch()


class SpaceTimeVisitor(AStarVisitor):
    def __init__(self, generating_graph, goal, limit=100, node_constraints=None, edge_constraints=None):
        self.timeless = generating_graph
        self.g = graph_tool.Graph(directed=True)
        self.g.vp['index'] = self.g.new_vertex_property('int')
        self.g.vp['t'] = self.g.new_vertex_property('int')
        self.g.ep['dist'] = self.g.new_edge_property('double')
        self.g.vp['cost'] = self.g.new_vertex_property('double')
        # self.g.vp['dist'] = self.g.new_vertex_property('double')

        self.target = goal
        for v in self.timeless.vertices():
            n = self.g.add_vertex()
            self.g.vp['index'][n] = int(v)
            self.g.vp['t'][n] = 0
        
        self.touched_e = self.g.new_edge_property('bool')
        self.touched_v = self.g.new_vertex_property('bool')
        self.limit=limit
        self.nc = node_constraints if node_constraints else {}
        self.ec = edge_constraints if edge_constraints else {}

    def get_node(self, t, index):
        for v in self.g.vertices():
            if self.g.vp['index'][v] == index and self.g.vp['t'][v] == t:
                return v
        return None

    def add_vertex_if_not_exists(self, t, index):
        if t in self.nc:
            if index in self.nc[t]:
                return None
        n = self.get_node(t, index)
        if n is not None:
            return n
        n = self.g.add_vertex()
        self.g.vp['index'][n] = index
        self.g.vp['t'][n] = t
        self.g.vp['cost'][n] = np.inf
        self.g.vp['dist'][n] = np.inf
        return n

    def add_edge_if_feasible(self, t, v1, v2):
        """add an edge to the graph from v1 to v2 iff. the edge is not forbidden"""

        i1 = self.g.vp['index'][v1]
        i2 = self.g.vp['index'][v2]

        if i1 == i2:
            e = self.g.add_edge(v1, v2)
            self.g.ep['dist'][e] = .1
            return e

        if t in self.ec:
            if (i1, i2) in self.ec[t]:
                return None
            if (i2, i1) in self.ec[t]:
                return None
        
        e = self.g.add_edge(v1, v2)
        self.g.ep['dist'][e] = self.timeless.ep['dist'][self.timeless.edge(i1, i2)]

        return e


    def add_outgoing_edges(self, v):
        index = self.g.vp['index'][v]
        t = self.g.vp['t'][v]
        for n in self.timeless.vertex(index).all_neighbors():
            vout = self.add_vertex_if_not_exists(t+1, int(n))
            if vout is not None:
                self.add_edge_if_feasible(t, v, vout)
        # add edge to self node (waiting action)
        
        vout = self.add_vertex_if_not_exists(t+1, index)
        if vout is not None:
            e = self.g.add_edge(v, vout)
            self.g.ep['dist'][e] = .1

    def discover_vertex(self, u):
        self.touched_v[u] = True
        if self.g.vp['t'][u] > self.limit:
            self.timed_target_node = None
            raise StopSearch()
        # add nodes and edges going out of this node
        self.add_outgoing_edges(u)

    def examine_edge(self, e):
        self.touched_e[e] = True
    
    def edge_relaxed(self, e):
        if self.g.vp['index'][e.target()] == int(self.target):
            self.timed_target_node = e.target()
            raise StopSearch()


def check_node_constraints(g, v, nc):
    t = g.vp['t'][v]
    i = g.vp['index'][v]
    if t not in nc:
        return False
    return i in nc[t]


def pred_to_list(g, pred, start, goal):
    if goal is None:
        print("goal was NONE, this means no path was found")
        return [start]
    p = goal
    l = [p]
    g.vp['visited'] = g.new_vertex_property("bool")
    while p != start:
        if p is None:
            print(l)
            return l
        if pred[p] == p:
            break
        p = pred[p]
        l.append(p)
    l.reverse()
    return l


def add_edge_constraints(edge_constraints, path):
    """append the constraints resulting from particular path to the list of existing constraints"""
    ec = {t: [e, tuple(reversed(e))] for t, e in enumerate(zip(path[:-1], path[1:]))}
    if edge_constraints is None:
        return ec
    for k, v in ec.items():
        if k in edge_constraints:
            edge_constraints[k] += v
        else:
            edge_constraints[k] = v
        
    return edge_constraints


def add_node_constraints(node_constraints, path:list, limit=0) -> dict:
    """append the constraints resulting from particular path to the list of existing constraints"""
    nc = {t: [n] for t, n in enumerate(path)}
    # if there is a limit on path length, we want to block the node where the robot is sitting after the goal is finished
    if len(path) < limit:
        nc.update( {t: [path[-1]] for t in range(len(path), limit)} )
    if node_constraints is None:
        return nc
    for k, v in nc.items():
        if k in node_constraints:
            node_constraints[k] += v
        else:
            node_constraints[k] = v
        
    return node_constraints


def prioritized_plans(g, start_goal, edge_constraints=None, node_constraints=None, limit=10):
    """compute a set of paths for multiple agents in the same graph.
    first agent is planned first, constraints are created for the remaining agents
    start_goal -- [(start, goal) for agent in agents]"""
    # plan first path with A-Star
    # g.set_edge_filter(g.ep['traversable'])
    # g.set_vertex_filter(g.vp['traversable'])
    paths = []
    for sn, gn in start_goal:
        paths.append(find_constrained_path(g, sn, gn, edge_constraints=edge_constraints, node_constraints=node_constraints, limit=limit))
        edge_constraints = add_edge_constraints(edge_constraints, paths[-1])
        node_constraints = add_node_constraints(node_constraints, paths[-1], limit=limit)
    
    return paths


def find_constrained_path(g, sn, gn, edge_constraints=None, node_constraints=None, limit=10):
    if edge_constraints is None and node_constraints is None:
        return find_path_astar(g, sn, gn)

    nv = SpaceTimeVisitor(g, gn, node_constraints=node_constraints, edge_constraints=edge_constraints, limit=limit)
    _, pred = astar_search(nv.g,
        sn,
        nv.g.ep['dist'],
        nv,
        implicit=True,
        heuristic=lambda v: np.linalg.norm(np.array(g.vp['center'][nv.g.vp['index'][v]]) - g.vp['center'][gn]),
        cost_map=nv.g.vp['cost'],
        dist_map=nv.g.vp['dist']
    )

    l = pred_to_list(nv.g, pred, sn, nv.timed_target_node)
    return [nv.g.vp["index"][v] for v in l]


def find_path_astar(g, sn, gn):
    """find shortest path through graph g with a* algorithm"""
    visitor = NavVisitor(g.new_vertex_property("bool"), g.new_edge_property("bool"), gn)
    _, pred = astar_search(g,
        sn,
        g.ep['dist'],
        visitor,
        heuristic=lambda v: np.linalg.norm(np.array(g.vp['center'][v]) - np.array(g.vp['center'][gn]))
    )
    return pred_to_list(g, pred, sn, gn)


def pad_path(path: list, limit=10) -> list:
    return path + [path[-1] for _ in range(limit- len(path))]

def compute_node_conflicts(paths: list, limit:int=10) -> list:
    node_occupancy = compute_node_occupancy(paths, limit=limit)
    
    conflicts = []
    for (t, node), agents in node_occupancy.items():
        if len(agents) > 1:
            conflicts.append((CBSConstraint(time=t, node=node, agent=agent) for agent in agents))
    return conflicts

def compute_node_occupancy(paths: list, limit:int=10) -> dict:
    node_occupancy = {}
    for i, path in enumerate(paths):
        for t, node in enumerate(pad_path(path, limit=limit)):
            if (t, node) not in node_occupancy:
                node_occupancy[t, node] = [i]
            else:
                node_occupancy[t, node] += [i]
    return node_occupancy


def compute_edge_conflicts(paths, limit=10):
    node_occupancy = compute_node_occupancy(paths, limit=limit)
    
    conflicts = []
    for i, path in enumerate(paths):
        for t, node in enumerate(path):
            if t < 1:
                continue
            if (t-1, node) in node_occupancy.keys():
                if node_occupancy[t-1, node] != i:
                    c = (t, node, i)
                    if t > 1:
                        conflicts.append( (CBSConstraint(time=t, node=node, agent=i), CBSConstraint(time=t-1, node=node, agent=node_occupancy[t-1,node][0])) )
                    else:
                        conflicts.append( (CBSConstraint(time=t, node=node, agent=i)) )
    return conflicts

def sum_of_cost(paths):
    if paths is None or None in paths:
        return np.inf
    return sum([len(p) for p in paths])

@dataclass(eq=True, frozen=True, init=True)
class CBSConstraint:
    agent:int
    time:int
    node:int


class CBSNode:
    def __init__(self, constraints:frozenset=frozenset()):
        self.children = ()
        self.fitness = None
        self.paths = None
        self.conflicts = None
        self.open = True

    def iter(self):
        yield self
        for child in self.children:
            child.iter()
    
    
class CBS:
    def __init__(self, g, start_goal, agent_constraints=None, limit=10):
        self.start_goal = start_goal
        self.g = g
        self.limit = limit
        self.root = CBSNode(constraints=agent_constraints)
        self.cache = {}
        self.agents = (i for i, _ in enumerate(start_goal))

    def run(self):
        self.cache = {}
        self.best = None
        done = True
        while not done:
            done = not self.step()
        return self.best

    def step(self):
            for node in self.root.iter():
                if node.open:
                    self.evaluate_node(node)
                    return True
            return False

    def evaluate_node(self, node):
        node.solution = []
        for agent in self.agents:
            # we have a cache, so paths with the same preconditions do not have to be calculated twice
            nc = frozenset(c for c in node.constraints if c.agent == agent)
            if nc not in self.cache:
                sn, gn = self.start_goal[agent]
                self.cache[nc] = find_constrained_path(self.g, sn, gn, node_constraints=nc, limit=self.limit)
            node.solution.append(self.cache[nc])

        node.fitness = sum_of_cost(node.solution)
        if node.fitness > len(node.solution) * self.limit:
            node.final = True
            node.open = False
            return
        node.conflicts = compute_node_conflicts(node.solution)
        if not len(node.conflicts):
            node.conflitcs = compute_edge_conflicts(node.solution)
        
        if len(node.conflicts):
            node.children = frozenset(CBSNode(constraints=c) for c in node.conflicts[0])
        else:
            if self.best is None or node.fitness < self.best.fitness:
                self.best = node
        node.open = False
        
    