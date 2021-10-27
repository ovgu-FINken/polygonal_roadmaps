from dataclasses import dataclass
import numpy as np
import networkx as nx
from networkx.algorithms.shortest_paths.weighted import _weight_function
from heapq import heappush, heappop
from itertools import count


def gen_example_graph(a, b):
    if b > a:
        a, b = b, a
    example = nx.Graph()
    for i in range(a):
        example.add_node(chr(i + ord('a')), pos=(i, 0))
    for i in range(a - 1):
        example.add_edge(chr(i + ord('a')), chr(i + ord('b')))
    for i in range(a, a + b):
        example.add_node(chr(i + ord('a')), pos=(i - a + (a - b) / 2, 1))
    for i in range(a, a + b - 1):
        example.add_edge(chr(i + ord('a')), chr(i + ord('b')))
    example.add_edge(chr(ord('a') + int((a - b) / 2)), chr(ord('a') + a))
    example.add_edge(chr(ord('a') - int((a - b) / 2) - 1 + a), chr(ord('a') + a + b - 1))

    for source, sink in example.edges():
        p1 = np.array(example.nodes()[source]['pos'])
        p2 = np.array(example.nodes()[sink]['pos'])
        example.edges()[source, sink]['dist'] = np.linalg.norm(p1 - p2)

    return example


@dataclass(eq=True, frozen=True, init=True)
class NodeConstraint:
    agent: int
    time: int
    node: int


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
        raise nx.NetworkXNoPath('no path could be found')
    return l


def pad_path(path: list, limit=10) -> list:
    return path + [path[-1] for _ in range(limit - len(path))]


def compute_node_conflicts(paths: list, limit: int = 10) -> list:
    node_occupancy = compute_node_occupancy(paths, limit=limit)

    conflicts = []
    for (t, node), agents in node_occupancy.items():
        if len(agents) > 1:
            conflicts.append(frozenset([NodeConstraint(time=t, node=node, agent=agent) for agent in agents]))
    return frozenset(conflicts)


def compute_node_occupancy(paths: list, limit: int = 10) -> dict:
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
            if (t - 1, node) in node_occupancy.keys():
                j = node_occupancy[t - 1, node][0]
                if j != i:
                    if t > 1:
                        conflicts.append(frozenset([NodeConstraint(time=t, node=node, agent=i),
                                         NodeConstraint(time=t - 1, node=node, agent=j)]))
                    else:
                        conflicts.append(frozenset([NodeConstraint(time=t, node=node, agent=i)]))
    return frozenset(conflicts)


def spatial_astar(G, source, target):
    def heuristic(u, v):
        return np.linalg.norm(np.array(G.nodes()[u]['pos']) - np.array(G.nodes()[v]['pos']))
    return nx.astar_path(G, source=source, target=target, heuristic=heuristic, weight='dist')


def compute_cost(G, target, limit=100):
    return nx.single_source_shortest_path_length(G, target, cutoff=limit)


def temporal_node_list(G, limit, node_constraints):
    nodes = G.nodes()
    nodelist = []
    for t in range(limit):
        nodelist += [(f'n{n}t{t}', {'n': n, 't': t}) for n in nodes if (n, t) not in node_constraints]
    return nodelist


def spacetime_astar(G, source, target, cost, limit=100, node_constraints=None):
    # we search the path one time with normal A*, so we know that all nodes exist and are connected
    if not node_constraints:
        spatial_path = spatial_astar(G, source, target)
        if len(spatial_path) > limit:
            raise nx.NetworkXNoPath()
        return spatial_astar(G, source, target)

    def heuristic(u):
        return cost[u]

    push = heappush
    pop = heappop
    weight = _weight_function(G, "dist")

    # The queue stores priority, node, cost to reach, and parent.
    # Uses Python heapq to keep in priority order.
    # Add a counter to the queue to prevent the underlying heap from
    # attempting to compare the nodes themselves. The hash breaks ties in the
    # priority and is guaranteed unique for all nodes in the graph.
    c = count()
    nodes = {f'n{source}t0': {'n': source, 't': 0, 'cost': 0, 'parent': None}}
    queue = [(0, next(c), f'n{source}t0', 0, None)]

    # Maps enqueued nodes to distance of discovered paths and the
    # computed heuristics to target. We avoid computing the heuristics
    # more than once and inserting the node into the queue too many times.
    enqueued = {}
    # Maps explored nodes to parent closest to the source.
    explored = {}

    while queue:
        # Pop the smallest item from queue.
        _, __, curnode, dist, parent = pop(queue)
        if nodes[curnode]['n'] == target:
            path = [curnode]
            node = parent
            while node is not None:
                path.append(node)
                node = explored[node]
            path.reverse()
            return [nodes[n]['n'] for n in path]

        if curnode in explored:
            # Do not override the parent of starting node
            if explored[curnode] is None:
                continue

            # Skip bad paths that were enqueued before finding a better one
            qcost, h = enqueued[curnode]
            if qcost < dist:
                continue

        explored[curnode] = parent

        # expand neighbours and add edges
        t = nodes[curnode]['t'] + 1
        node = nodes[curnode]['n']

        for neighbor, w in G[node].items():
            if (neighbor, t) in node_constraints:
                continue

            ncost = dist + weight(node, neighbor, w)
            t_neighbor = f'n{neighbor}t{t}'
            if t_neighbor in enqueued:
                qcost, h = enqueued[t_neighbor]
                # if qcost <= ncost, a less costly path from the
                # neighbor to the source was already determined.
                # Therefore, we won't attempt to push this neighbor
                # to the queue
                if qcost <= ncost:
                    continue
            else:
                h = heuristic(neighbor)
            nodes[t_neighbor] = {'n': neighbor, 't': t}
            enqueued[t_neighbor] = ncost, h
            push(queue, (ncost + h, next(c), t_neighbor, ncost, curnode))

        if (node, t) not in node_constraints:
            neighbor = node

            ncost = dist + 1
            t_neighbor = f'n{neighbor}t{t}'
            if t_neighbor in enqueued:
                qcost, h = enqueued[t_neighbor]
                # if qcost <= ncost, a less costly path from the
                # neighbor to the source was already determined.
                # Therefore, we won't attempt to push this neighbor
                # to the queue
                if qcost <= ncost:
                    continue
            else:
                h = heuristic(neighbor)
            nodes[t_neighbor] = {'n': neighbor, 't': t}
            enqueued[t_neighbor] = ncost, h
            push(queue, (ncost + h, next(c), t_neighbor, ncost, curnode))

    raise nx.NetworkXNoPath(f"Node {target} not reachable from {source}")


def sum_of_cost(paths, graph=None):
    if paths is None or None in paths or [] in paths:
        return np.inf
    if graph is None:
        return sum([len(p) for p in paths])
    cost = 0
    for path in paths:
        for n1, n2 in zip(path[:-1], path[1:]):
            if n1 == n2:
                cost += 0.1
            else:
                cost += graph.edges()[n1, n2]['dist']
    return cost


class CBSNode:
    def __init__(self, constraints: frozenset = None):
        self.children = ()
        self.fitness = np.inf
        self.valid = False
        self.paths = None
        self.conflicts = None  # conflicts are found after plannig
        self.final = None
        self.solution = None
        self.constraints = constraints  # constraints are used for planning
        if self.constraints is None:
            self.constraints = frozenset()
        self.open = True

    def __iter__(self):
        yield self
        for child in self.children:
            yield from child

    def __str__(self):
        return f'{self.constraints}::[ {",".join([str(x) for x in self.children])} ]'

    def tuple_repr(self):
        conflicts = len(self.conflicts) if self.conflicts else 0
        constraints = len(self.constraints) if self.constraints else 0
        return self.fitness, conflicts, constraints

    def __eq__(self, other):
        return self.tuple_repr() == other.tuple_repr()

    def __neq__(self, other):
        return self.tuple_repr() != other.tuple_repr()

    def __lt__(self, other):
        return self.tuple_repr() < other.tuple_repr()

    def __le__(self, other):
        return self.tuple_repr() <= other.tuple_repr()

    def __gt__(self, other):
        return self.tuple_repr() > other.tuple_repr()

    def __ge__(self, other):
        return self.tuple_repr() >= other.tuple_repr()

    def shorthand(self):
        result = "[constraints::"
        d = {}
        for x in self.constraints:
            if x.agent not in d:
                d[x.agent] = [(x.time, x.node)]
            else:
                d[x.agent].append((x.time, x.node))
        for a, group in d.items():
            result += f"agent {a}::{{"
            stuff = [f'(t: {t}, n: {n})' for t, n in sorted(group)]
            result += ",".join(stuff)

            result += "}"
        result += "]"
        if not self.conflicts:
            return result
        return result + f'conflicts : {len(self.conflicts)}'

    def graph(self):
        graph = nx.DiGraph()
        for i, x in enumerate(self):
            graph.add_node(
                i,
                label=x.shorthand(),
                weight=len(x.constraints),
                fitness=x.fitness,
                valid=x.valid,
                solution=str(x.solution),
                conflicts=str(x.conflicts),
                open=x.open
            )
            x._index = i
        for x in self:
            for y in x.children:
                graph.add_edge(x._index, y._index)
        return graph


class CBS:
    def __init__(self, g, start_goal, agent_constraints=None, limit=10, max_iter=10000):
        self.start_goal = start_goal
        self.g = g
        self.limit = limit
        self.root = CBSNode(constraints=agent_constraints)
        self.cache = {}
        self.agents = tuple([i for i, _ in enumerate(start_goal)])
        self.cache = {}
        self.costs = {agent: compute_cost(self.g, start_goal[agent][1], limit=limit) for agent in self.agents}
        self.best = None
        self.max_iter = max_iter
        self.iteration_counter = 0
        self.duplicate_cache = set()
        self.duplicate_counter = 0

    def setup(self):
        self.cache = {}
        self.best = None
        self.open = []
        self.evaluate_node(self.root)
        self.push(self.root)

    def push(self, node):
        heappush(self.open, node)

    def pop(self):
        node = heappop(self.open)
        node.open = False
        return node

    def run(self):
        self.setup()
        for _ in range(self.max_iter):
            if not self.step():
                break
        if self.best is None:
            raise nx.NetworkXNoPath()
        return self.best.solution

    def step(self):
        if self.open:
            self.iteration_counter += 1
            node = self.pop()
            for child in node.children:
                child = self.evaluate_node(child)
                child.open = False
                self.push(child)
            return True
        # return false if all nodes are already closed
        return False

    def evaluate_node(self, node):
        node.solution = []
        for agent in self.agents:
            # we have a cache, so paths with the same preconditions do not have to be calculated twice
            nc = frozenset([(c.node, c.time) for c in node.constraints if c.agent == agent])
            if nc not in self.cache:
                sn, gn = self.start_goal[agent]
                try:
                    self.cache[agent, nc] = spacetime_astar(
                        self.g, sn, gn, self.costs[agent], node_constraints=nc, limit=self.limit)
                except nx.NetworkXNoPath:
                    self.cache[agent, nc] = None
            node.solution.append(self.cache[agent, nc])

        node.fitness = sum_of_cost(node.solution, graph=self.g)
        if node.fitness > len(self.agents) * self.limit:
            # if no valid solution exists, this node is final
            node.final = True
            return node

        if self.best is not None and node.fitness > self.best.fitness:
            # if we add conflicts, the path only gets longer, hence this is not the optimal solution
            node.final = True
            return node

        node.conflicts = set()
        # node.conflicts = frozenset(set( conflict for conflict in compute_node_conflicts(node.solution)) | set(conflict for conflict in compute_edge_conflic
        for conflict in compute_node_conflicts(node.solution):
            node.conflicts |= conflict

        for conflict in compute_edge_conflicts(node.solution):
            node.conflicts |= conflict

        if not len(node.conflicts):
            node.valid = True
            node.final = True
            self.update_best(node)
            return node
        # filter out conflicts that are already in the constraints (happens if the conflict is after the goal is met)
        node.conflicts = set(c for c in node.conflicts if c not in node.constraints)

        children = []
        for constraint in node.conflicts:
            if constraint in node.constraints:
                # print(f"constraint: {constraint}, NC: {node.constraints}")
                continue
            constraints = frozenset({constraint} | node.constraints)
            if constraints not in self.duplicate_cache:
                children.append(CBSNode(constraints=frozenset({constraint} | node.constraints)))
                self.duplicate_cache.add(constraints)
            else:
                self.duplicate_counter += 1

        if len(children):
            node.children = tuple(children)
            return node

        node.final = True
        return node

    def update_best(self, node):
        if self.best is None or node.fitness < self.best.fitness:
            print(f'Found new best solution at iteration {self.iteration_counter}.')
            self.best = node
        return node


if __name__ == "__main__":
    pass
