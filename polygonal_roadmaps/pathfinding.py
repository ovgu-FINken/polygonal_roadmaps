from graph_tool.all import AStarVisitor, StopSearch, astar_search
import graph_tool
from dataclasses import dataclass
import numpy as np


@dataclass(eq=True, frozen=True, init=True)
class NodeConstraint:
    agent:int
    time:int
    node:int


class PathDoesNotExistException(ValueError):
    pass

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
        self.g.vp['dist'] = self.g.new_vertex_property('double')

        self.target = goal
        for v in self.timeless.vertices():
            n = self.g.add_vertex()
            self.g.vp['index'][n] = int(v)
            self.g.vp['t'][n] = 0
        
        self.touched_e = self.g.new_edge_property('bool')
        self.touched_v = self.g.new_vertex_property('bool')
        self.limit=limit
        self.nc = node_constraints if node_constraints else []
        self.timed_target_node = None
        #self.ec = edge_constraints if edge_constraints else {}

    def get_node(self, t, index):
        for v in self.g.vertices():
            if self.g.vp['index'][v] == index and self.g.vp['t'][v] == t:
                return v
        return None

    def node_violates_constraints(self, t, index):
        violated = [c for c in self.nc if c.node == index and c.time == t]
        return len(violated)

    def add_vertex_if_not_exists(self, t, index):
        if self.node_violates_constraints(t, index):
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
        
        # print(f'v1: {v1}:{i1}, v2: {v2}:{i2}, timeless: {self.timeless.edge(i1, i2)}')
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


def violated_node_constraints(g, v: int, nc: list) -> list:
    t = g.vp['t'][v]
    i = g.vp['index'][v]
    violated = []
    for constraint in nc:
        if constraint.t == t and constraint.node == i:
            violated.append(constraint)
    return violated


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
    if l[0] != start or l[-1] != goal:
        raise PathDoesNotExistException()
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
    if nv.timed_target_node is None:
        raise PathDoesNotExistException()

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
            conflicts.append(frozenset([NodeConstraint(time=t, node=node, agent=agent) for agent in agents]))
    return frozenset(conflicts)

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
                        conflicts.append( frozenset( [NodeConstraint(time=t, node=node, agent=i), NodeConstraint(time=t-1, node=node, agent=node_occupancy[t-1,node][0])] ) )
                    else:
                        conflicts.append( frozenset([NodeConstraint(time=t, node=node, agent=i)]) )
    return frozenset(conflicts)


def sum_of_cost(paths):
    if paths is None or None in paths or [] in paths:
        return np.inf
    return sum([len(p) for p in paths])


class CBSNode:
    def __init__(self, constraints:frozenset=None):
        self.children = ()
        self.fitness = np.inf
        self.paths = None
        self.conflicts = None # conflicts are found after plannig
        self.final = None
        self.constraints = constraints # constraints are used for planning
        if self.constraints is None:
            self.constraints = frozenset()
        self.open = True

    def __iter__(self):
        yield self
        for child in self.children:
            yield from child
    
    def __str__(self):
        return f'{self.constraints}::[ {",".join([str(x) for x in self.children])} ]'
    
    
class CBS:
    def __init__(self, g, start_goal, agent_constraints=None, limit=10):
        self.start_goal = start_goal
        self.g = g
        self.limit = limit
        self.root = CBSNode(constraints=agent_constraints)
        self.cache = {}
        self.agents = tuple([i for i, _ in enumerate(start_goal)])
        self.cache = {}
        self.best = None

    def run(self):
        self.cache = {}
        self.best = None
        done = False
        for _ in range(self.
            limit**len(self.agents)):
            if not self.step():
                break
        if self.best is None:
            raise PathDoesNotExistException
        return self.best.solution

    def step(self):
        for node in self.root:
            if node.open:
                node = self.evaluate_node(node)
                node.open = False
                return True
        # return false if all nodes are already closed
        return False

    def evaluate_node(self, node):
        node.solution = []
        for agent in self.agents:
            # we have a cache, so paths with the same preconditions do not have to be calculated twice
            nc = frozenset([c for c in node.constraints if c.agent == agent])
            if nc not in self.cache:
                sn, gn = self.start_goal[agent]
                try:
                    self.cache[agent, nc] = find_constrained_path(self.g, sn, gn, node_constraints=nc, limit=self.limit)
                except PathDoesNotExistException:
                    self.cache[agent, nc] = None
            node.solution.append(self.cache[agent, nc])

        node.fitness = sum_of_cost(node.solution)
        if node.fitness > len(self.agents) * self.limit:
            # if no valid solution exists, this node is final
            node.final = True
            return node
        
        if self.best is not None and node.fitness > self.best.fitness:
            # if we add conflicts, the path only gets longer, hence this is not the optimal solution
            node.final = True
            return node
            
        # this function calculates the children of the node and
        # in case we do not have node conflicts, we compute the edge conflicts
        node.conflicts = frozenset(compute_edge_conflicts(node.solution) | compute_edge_conflicts(node.solution))
    
        if not len(node.conflicts):
            node.final = True
            self.update_best(node)
            return node
        
        children = []
        for conflict in node.conflicts:
            for constraint in conflict:
                if constraint in node.constraints:
                    continue
                children.append(CBSNode(constraints = frozenset({constraint} | node.constraints)))
            if len(children):
                node.children = tuple(children)
                return node
            
        node.final = True
        return node

                
    def update_best(self, node):
        if self.best is None or node.fitness < self.best.fitness:
            self.best = node
        return node

if __name__ == "__main__":
    g = graph_tool.load_graph('test/resources/test_graph.xml')
    cbs = CBS(g, [(0,5), (5,0)], limit=100)
    result = cbs.run()