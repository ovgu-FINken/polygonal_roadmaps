from asyncio.log import logger
from dataclasses import dataclass
import numpy as np
import networkx as nx
from networkx.algorithms.shortest_paths.weighted import _weight_function
from heapq import heappush, heappop
from itertools import count
from functools import partial

import logging


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


def read_movingai_map(path):
    # read a map given with the movingai-framework
    with open(path) as map_file:
        lines = map_file.readlines()
    height = int("".join([d for d in lines[1] if d in list("0123456789")]))
    width = int("".join([d for d in lines[2] if d in list("0123456789")]))

    graph = nx.grid_2d_graph(width, height)
    for edge in graph.edges():
        graph.edges()[edge]['dist'] = 1
    graph.add_edges_from(
        [((x, y), (x + 1, y + 1)) for x in range(width - 1) for y in range(height - 1)],
        dist=np.sqrt(2)
    )
    graph.add_edges_from(
        [((x + 1, y), (x, y + 1)) for x in range(width - 1) for y in range(height - 1)],
        dist=np.sqrt(2)
    )
    data = lines[4:]
    blocked = list("@OTW")
    for i, row in enumerate(data):
        for j, pixel in enumerate(row[:-1]):
            if pixel in blocked:
                graph.remove_node((i, j))

    for node in graph.nodes():
        graph.nodes()[node]["pos"] = node
    return graph


@dataclass(eq=True, frozen=True, init=True)
class NodeConstraint:
    agent: int
    time: int
    node: int


def pred_to_list(g, pred, start, goal):
    if goal is None:
        logging.debug("goal was NONE, this means no path was found")
        return [start]
    p = goal
    path = [p]
    g.vp['visited'] = g.new_vertex_property("bool")
    while p != start:
        if p is None:
            logging.debug(path)
            return path
        if pred[p] == p:
            break
        p = pred[p]
        path.append(p)
    path.reverse()
    if path[0] != start or path[-1] != goal:
        raise nx.NetworkXNoPath('no path could be found')
    return path


def pad_path(path: list, limit: int = None) -> list:
    if limit is None:
        return path
    return path + [path[-1] for _ in range(limit - len(path))]


def compute_node_conflicts(paths: list, limit: int = None) -> list:
    node_occupancy = compute_node_occupancy(paths, limit=limit)

    conflicts = []
    for (t, node), agents in node_occupancy.items():
        if len(agents) > 1:
            conflicts.append(frozenset([NodeConstraint(time=t, node=node, agent=agent) for agent in agents]))
    return frozenset(conflicts)


def compute_node_occupancy(paths: list, limit: int = None) -> dict:
    node_occupancy = {}
    for i, path in enumerate(paths):
        for t, node in enumerate(pad_path(path, limit=limit)):
            if (t, node) not in node_occupancy:
                node_occupancy[t, node] = [i]
            else:
                node_occupancy[t, node] += [i]
    return node_occupancy


@dataclass(frozen=True, eq=True, init=True)
class Conflict:
    k: int
    conflicting_agents: frozenset

    def generate_constraints(self):
        """ generate a set of constraints for each conflicting agent, which resolves the conflict
        i.e., one agent can stay, all others have to move to resolve the conflict
        in a k=0 conflict, this means for each agent a set containing the other agents
        in a k!=0 conflict this means either the agent at time t moves or all agents at t-k move
        """
        if self.k == 0:
            return {self.conflicting_agents - {agent} for agent in self.conflicting_agents}
        t_agent = max(self.conflicting_agents, key=lambda x: x.time)
        return {frozenset(self.conflicting_agents - {t_agent}), frozenset({t_agent})}


def compute_k_robustness_conflicts(paths, limit: int = None, k: int = 0, node_occupancy: dict = None):
    if node_occupancy is None:
        node_occupancy = compute_node_occupancy(paths, limit=limit)

    conflicts = []
    for i, path in enumerate(paths):
        for t, node in enumerate(path):
            if t < k:
                continue
            if (t - k, node) not in node_occupancy:
                continue
            # compute set of agents occupying the node at time t-k, without i
            conflicting_agents = set(node_occupancy[t - k, node])
            conflicting_agents.discard(i)
            # if the set of conflicting agents is empty, there is no conflict
            if not len(conflicting_agents):
                continue
            # we now have established that there is a conflict
            # this means the agent itself could move to resolve the conflict
            constraints = {
                NodeConstraint(time=t, node=node, agent=i)
            }
            # or all the other agents could move to resolve the conflict
            # however there is a special case: if t - k == 0 the other agents can not move
            # (because the starting position is fixed)
            if t != k:
                constraints |= {NodeConstraint(time=t - k, node=node, agent=j) for j in conflicting_agents}
            conflicts.append(Conflict(k=k, conflicting_agents=frozenset(constraints)))
    return set(conflicts)


def compute_all_k_conflicts(solution, limit: int = None, k=1):
    """compute the conflicts present in a solution
    Each conflict is a set of triples {(agent, time, node)
    - out of each conflict at least one agent has to move away to resolve the conflict
    If there is no conflict the solution is valid
    Conflicts are used to generate constraints
    """
    node_occupancy = compute_node_occupancy(solution, limit=limit)
    conflicts = [compute_k_robustness_conflicts(solution,
                                                limit=limit,
                                                k=ik,
                                                node_occupancy=node_occupancy)
                 for ik in range(k + 1)]
    return frozenset().union(*conflicts)


def spatial_astar(G, source, target, weight=None):
    def heuristic(u, v):
        return np.linalg.norm(np.array(G.nodes()[u]['pos']) - np.array(G.nodes()[v]['pos']))
    return nx.astar_path(G, source=source, target=target, heuristic=heuristic, weight=weight)


def compute_cost(G, target, weight=None):
    return nx.shortest_path_length(G, target=target, weight=weight)


def temporal_node_list(G, limit, node_constraints):
    nodes = G.nodes()
    nodelist = []
    for t in range(limit):
        nodelist += [(f'n{n}t{t}', {'n': n, 't': t}) for n in nodes if (n, t) not in node_constraints]
    return nodelist


def spacetime_astar(G, source, target, heuristic=None, limit=100, node_constraints=None):
    # we search the path one time with normal A*, so we know that all nodes exist and are connected
    if not node_constraints:
        spatial_path = spatial_astar(G, source, target)
        if len(spatial_path) > limit:
            raise nx.NetworkXNoPath()
        return spatial_path, sum_of_cost([spatial_path], graph=G, weight="weight")

    def default_heuristic(u):
        return nx.shortest_path_length(G, source=u, target=target, weight="weight")
    if heuristic is None:
        heuristic = default_heuristic

    push = heappush
    pop = heappop
    weight_fn = _weight_function(G, "weight")

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
            return spacetime_astar_calculate_return_path(nodes, explored, curnode, parent), dist

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
            ncost = dist + weight_fn(node, neighbor, w)
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

            ncost = dist + 1.0001
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


def spacetime_astar_calculate_return_path(nodes, explored, curnode, parent):
    path = [curnode]
    node = parent
    while node is not None:
        path.append(node)
        node = explored[node]
    path.reverse()
    return [nodes[n]['n'] for n in path]


def solution_valid(solution: list) -> bool:
    if solution is None or None in solution or [] in solution:
        return False
    if not len(solution):
        return False
    return True


def sum_of_cost(paths, graph=None, weight=None) -> float:
    if not solution_valid(paths):
        return np.inf
    if graph is None or weight is None:
        return sum([len(p) for p in paths])
    cost = 0
    for path in paths:
        n1 = path[0]
        for n2 in path[1:]:
            if n1 == n2:
                cost += 1.0001
            else:
                cost += graph.edges()[n1, n2][weight]
            n1 = n2
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

    def heuristic(self):
        if not self.children:
            return self.fitness
        return min(child.heuristic() for child in self.children)

    def tuple_repr(self):
        conflicts = len(self.conflicts) if self.conflicts else 0
        return self.heuristic(), conflicts, self.fitness

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


def compute_normalized_weight(g, weight):
    if weight is not None:
        max_dist = max(nx.get_edge_attributes(g, weight).values())
        # add a weight attribute, which normalizes the distance between nodes by the maximum distane
        # the maximum distance will get a weight of 1, waiting action will also get the weight of 1+eps
        for n1, n2 in g.edges():
            g.edges()[n1, n2]["weight"] = g.edges()[n1, n2][weight] / max_dist
    else:
        for n1, n2 in g.edges():
            g.edges()[n1, n2]['weight'] = 1.0001


class CBS:
    def __init__(self,
                 g,
                 start_goal,
                 weight=None,
                 agent_constraints=None,
                 limit=10, max_iter=10000,
                 pad_paths=True,
                 k_robustness=1):
        self.start_goal = start_goal
        for start, goal in start_goal:
            if start not in g.nodes() or goal not in g.nodes():
                raise nx.NodeNotFound()
        self.pad_paths = pad_paths
        self.g = g
        compute_normalized_weight(self.g, weight)
        self.limit = limit
        self.root = CBSNode(constraints=agent_constraints)
        self.cache = {}
        self.agents = tuple([i for i, _ in enumerate(start_goal)])
        self.costs = {agent: {} for agent in self.agents}
        self.best = None
        self.max_iter = max_iter
        self.iteration_counter = 0
        self.duplicate_cache = set()
        self.duplicate_counter = 0
        self.k_robustness = k_robustness

    def heuristic(self, node, agent=None):
        if node not in self.costs[agent]:
            _, goal = self.start_goal[agent]
            self.costs[agent][goal] = 0
            path = spatial_astar(self.g, goal, node, weight="weight")
            assert(path[-1] == node)
            cost = 0
            for n1, n2 in zip(path[:-1], path[1:]):
                cost += self.g.edges()[n1, n2]["weight"]
                if n2 not in self.costs[agent]:
                    self.costs[agent][n2] = cost
        return self.costs[agent][node]

    def setup(self):
        self.cache = {}
        self.best = None
        self.open = []
        self.evaluate_node(self.root)
        if self.best is None:
            self.repair_node_solution(self.root)
        self.push(self.root)

    def push(self, node):
        # nodes are sortable, the sort-order of those heaps lead to the order in the heap
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
        return self.best

    def step(self):
        if not self.open:
            return False
        # return false if all nodes are already closed
        self.iteration_counter += 1
        node = self.pop()
        if self.iteration_counter % 500 == 0:
            logging.info(f"""{self.iteration_counter / self.max_iter * 100}%:
    h = {node.heuristic()},
    f = {node.fitness},
    cost = {len(self.cache)},
    conflicts = {len(node.conflicts)},
    nodes = {len(self.open)}""")
        for child in node.children:
            child = self.evaluate_node(child)
            child.open = False
            if not child.final:
                self.push(child)
        return True

    def repair_node_solution(self, node):
        # implementation: quick and dirty
        limit = self.limit if self.pad_paths else None
        constraints = set(node.constraints)
        for agent in self.agents:
            solution, _ = self.compute_node_solution(CBSNode(constraints), max_agents=agent + 1)
            if not solution_valid(solution):
                # unsuccessful
                logging.debug("Repairing Solution is not successful")
                return
            for t, n in enumerate(pad_path(solution[agent], limit=limit)):
                # add constraints to the paths of all the following agents
                for k in range(self.k_robustness + 1):
                    constraints |= {NodeConstraint(agent=ti, time=t + k, node=n) for ti in self.agents[agent:]}
            logging.debug(f"prioritized plan: {agent}")
        fake_node = CBSNode(frozenset(constraints))
        fake_node.solution, fake_node.fitness = self.compute_node_solution(fake_node)
        fake_node.final = True
        fake_node.valid = True
        if self.best is not None:
            logging.debug(f"before: {self.best.fitness}")
        self.update_best(fake_node)
        logging.debug(f"after: {self.best.fitness}")
        node.children = (*node.children, fake_node)

    def compute_node_solution(self, node, max_agents=None):
        solution = []
        if max_agents is None:
            max_agents = len(self.agents)
        # sum of cost
        soc = 0
        for agent in self.agents[:max_agents]:
            # we have a cache, so paths with the same preconditions do not have to be calculated twice
            nc = frozenset([(c.node, c.time) for c in node.constraints if c.agent == agent])
            if (agent, nc) not in self.cache:
                sn, gn = self.start_goal[agent]
                try:
                    self.cache[agent, nc] = spacetime_astar(
                        self.g, sn, gn, heuristic=partial(self.heuristic, agent=agent), node_constraints=nc, limit=self.limit)
                except nx.NetworkXNoPath:
                    logger.warn("No path for agent")
                    self.cache[agent, nc] = None, np.inf
            path, cost = self.cache[agent, nc]
            solution.append(path)
            soc += cost
        return solution, soc

    def compute_node_fitness(self, node):
        return sum_of_cost(node.solution, graph=self.g, weight="weight")

    def evaluate_node(self, node):
        if node.final:
            return node

        if node.solution is None:
            node.solution, node.fitness = self.compute_node_solution(node)

        if not solution_valid(node.solution):
            # if no valid solution exists, this node is final
            node.final = True
            node.valid = False
            logging.debug("set node to final because of high fitness value")
            return node

        assert(node.fitness != np.inf)

        if self.best is not None and node.heuristic() >= self.best.fitness:
            # if we add conflicts, the path only gets longer, hence this is not the optimal solution
            node.final = True
            logging.debug(f"set node to final because of best fitness value: {node.fitness} >= {self.best.fitness}")
            return node

        limit = self.limit if self.pad_paths else None
        node.conflicts = compute_all_k_conflicts(node.solution, limit=limit, k=self.k_robustness)

        if not len(node.conflicts):
            logging.debug("set node to final because no conflicts")

            node.valid = True
            node.final = True
            self.update_best(node)
            return node
        return self.expand_node(node)

    def expand_node(self, node):
        heuristic = self.compute_node_conflict_heurstic(node)
        # get one of the conflicts, to create constraints from
        # each conflict must be resolved in some way, so we pick the most expensive conflict first
        # by picking the most expensive conflict, we generate less solutions with good fitness
        working_conflict = sorted(node.conflicts, key=lambda x: heuristic[x], reverse=True)[0]
        children = []
        for constraints in working_conflict.generate_constraints():
            if node.constraints.issuperset(constraints):
                continue
            constraints = frozenset(constraints | node.constraints)
            if constraints not in self.duplicate_cache:
                self.duplicate_cache.add(constraints)
                child = CBSNode(constraints=constraints)
                child.solution, child.fitness = self.compute_node_solution(child)
                # do not consider children without valid solution
                if not solution_valid(child.solution):
                    continue
                children.append(child)
            else:
                self.duplicate_counter += 1

        node.children = tuple(children)
        if len(node.children):
            return node
        logging.warn("no children added, node is not valid")
        logging.debug(f"{node.conflicts}, {node.constraints}")
        node.final = True
        return node

    def update_best(self, node):
        if not node.valid:
            return self.best
        # in case we already have a better solution, keep it
        if self.best is not None and node.fitness >= self.best.fitness:
            return self.best

        # this is the best solution we found
        self.best = node
        logging.info(f"found new best at iteration {self.iteration_counter}, fitness: {self.best.fitness}")

        # remove all the solution not as good as best from the open list
        self.open = [x for x in self.open if x.heuristic() < self.best.fitness]
        return node

    def compute_node_conflict_heurstic(self, node):
        h = {}
        for conflict in node.conflicts:
            min_cost = np.inf
            for constraints in conflict.generate_constraints():
                _, cost = self.compute_node_solution(CBSNode(constraints=node.constraints | constraints))
                min_cost = min(cost, min_cost)
            h[conflict] = min_cost
        return h


def prioritized_plans(graph, start_goal, constraints=frozenset(), limit=10, pad_paths=True):
    solution = []
    compute_normalized_weight(graph, "dist")
    for start, goal in start_goal:
        if pad_paths:
            node_occupancy = compute_node_occupancy(solution, limit=limit)
        else:
            node_occupancy = compute_node_occupancy(solution, limit=None)
        constraints = set()
        for t, node in node_occupancy.keys():
            if t > 0:
                constraints.add((node, t))
                constraints.add((node, t + 1))
        logging.debug(constraints)
        path, _ = spacetime_astar(graph, start, goal,
                                  limit=limit, node_constraints=constraints)
        solution.append(path)
    return solution


if __name__ == "__main__":
    pass
