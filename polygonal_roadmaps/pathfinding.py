from asyncio.log import logger
from dataclasses import dataclass
import numpy as np
import networkx as nx
from networkx.algorithms.shortest_paths.weighted import _weight_function
from heapq import heappush, heappop
from itertools import count
from functools import partial
import copy

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


def remove_edge_if_exists(g: nx.Graph, u, v) -> None:
    if g.has_edge(u, v):
        g.remove_edge(u, v)


def remove_node_if_exists(g: nx.Graph, v) -> None:
    if g.has_node(v):
        g.remove_node(v)


def read_movingai_map(path):
    # read a map given with the movingai-framework
    with open(path) as map_file:
        lines = map_file.readlines()
    height = int("".join([d for d in lines[1] if d in list("0123456789")]))
    width = int("".join([d for d in lines[2] if d in list("0123456789")]))

    graph = nx.grid_2d_graph(height, width)
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
                remove_node_if_exists(graph, (i, j))
                remove_edge_if_exists(graph, (i + 1, j), (i, j + 1))
                remove_edge_if_exists(graph, (i - 1, j), (i, j + 1))
                remove_edge_if_exists(graph, (i + 1, j), (i, j - 1))
                remove_edge_if_exists(graph, (i - 1, j), (i, j - 1))

    for node in graph.nodes():
        graph.nodes()[node]["pos"] = node
    return graph, width, height, data


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
    if path is None or not len(path):
        return []
    if limit > len(path):
        return path + [path[-1] for _ in range(limit - len(path))]
    if limit < len(path):
        return path[:limit]
    return path


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


def compute_solution_robustness(solution, limit: int = None):
    maximum = max([len(path) for path in solution])
    for k in range(maximum):
        conflits = compute_all_k_conflicts(solution, limit=limit, k=k)
        if len(conflits):
            return k - 1
    return maximum


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
    def spatial_heuristic(u, v):
        return np.linalg.norm(np.array(G.nodes()[u]['pos']) - np.array(G.nodes()[v]['pos']))
    return nx.astar_path(G, source=source, target=target, heuristic=spatial_heuristic, weight=weight)


def compute_cost(G, target, weight=None):
    return nx.shortest_path_length(G, target=target, weight=weight)


def temporal_node_list(G, limit, node_constraints):
    nodes = G.nodes()
    nodelist = []
    for t in range(limit):
        nodelist += [(f'n{n}t{t}', {'n': n, 't': t}) for n in nodes if (n, t) not in node_constraints]
    return nodelist


def spacetime_astar(G, source, target, spacetime_heuristic=None, limit=100, node_constraints=None, wait_action_cost=1.0001):
    # we search the path one time with normal A*, so we know that all nodes exist and are connected
    if not node_constraints:
        spatial_path = spatial_astar(G, source, target)
        if len(spatial_path) > limit:
            raise nx.NetworkXNoPath()
        return spatial_path, sum_of_cost([spatial_path], graph=G, weight="weight")

    def true_cost_heuristic(u):
        return nx.shortest_path_length(G, source=u, target=target, weight="weight")
    if spacetime_heuristic is None:
        spacetime_heuristic = true_cost_heuristic

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
                h = spacetime_heuristic(neighbor)
            nodes[t_neighbor] = {'n': neighbor, 't': t}
            enqueued[t_neighbor] = ncost, h
            push(queue, (ncost + h, next(c), t_neighbor, ncost, curnode))

        if (node, t) not in node_constraints:
            neighbor = node

            ncost = dist + wait_action_cost
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
                h = spacetime_heuristic(neighbor)
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
    if solution is None or None in solution:
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


def nx_shortest(*args, **kwargs):
    try:
        return nx.shortest_path_length(*args, **kwargs)
    except nx.NetworkXNoPath:
        logging.info('no path')
    # inf, but not inf (avoids NAN-problems)
    return 1e100


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


class SpaceTimeAStarCache:
    """ Cache for paths (with constraints) and the heurstic precomputed by spatial A-star """
    def __init__(self, graph, costs=None, kwargs=None):
        self.kwargs = kwargs
        self.costs = {}
        self.cache = {}
        self.graph = graph

    def cbs_heurstic(self, node, goal=None):
        if goal not in self.costs:
            self.costs[goal] = {goal: 0}
        if node not in self.costs[goal]:
            path = spatial_astar(self.graph, goal, node, weight="weight")
            assert(path[-1] == node)
            cost = 0
            for n1, n2 in zip(path[:-1], path[1:]):
                cost += self.graph.edges()[n1, n2]["weight"]
                if n2 not in self.costs[goal]:
                    self.costs[goal][n2] = cost
        return self.costs[goal][node]

    def reset(self, state_only=False):
        """"reset the cache
        if state_only=True is passed, the cost to the goals is not reset.
        """
        self.costs = {}
        if not state_only:
            self.cache = {}

    def get_path(self, start, goal, node_constraints):
        """try to look up a path from start to goal in the cache with specific contstraints
        if the path was searched before, we will return the path without new computation
        """
        if start is None:
            return [], 0
        if (start, goal, node_constraints) in self.cache:
            return self.cache[start, goal, node_constraints]
        try:
            self.cache[start, goal, node_constraints] = spacetime_astar(
                self.graph,
                start,
                goal,
                spacetime_heuristic=partial(self.cbs_heurstic, goal=goal),
                node_constraints=node_constraints,
                **self.kwargs)
        except nx.NetworkXNoPath:
            logger.warning("No path for agent")
            self.cache[start, goal, node_constraints] = None, np.inf
        return self.cache[start, goal, node_constraints]


def decision_function(qualities, method=None):
    """compute a collective decison based on qualities of the options
    the default behaviour is to use the direct comparison method, which uses the option with the maximum of all quality values
    returns the index of the best quality option"""
    if method is None or method == 'direct_comparison':
        columns = np.max(np.array(qualities), axis=1, where=~np.isnan(np.array(qualities)), initial=-np.inf)
        return np.argmax(columns)
    elif method == 'random':
        return np.random.choice([i for i, _ in enumerate(qualities[0])])
    else:
        raise NotImplementedError(f"{method} is not available as a decision function")

    return 0


class CBS:
    def __init__(self,
                 g,
                 start_goal,
                 weight=None,
                 agent_constraints=None,
                 limit=10, max_iter=10000,
                 pad_paths=True,
                 k_robustness=1,
                 wait_action_cost=1.0001,
                 discard_conflicts_beyond=None):
        self.start_goal = start_goal
        for start, goal in start_goal:
            if start not in g.nodes() or goal not in g.nodes():
                raise nx.NodeNotFound()
        self.pad_paths = pad_paths
        self.g = g
        compute_normalized_weight(self.g, weight)
        self.limit = limit
        self.agent_constraints = agent_constraints
        self.root = CBSNode(constraints=self.agent_constraints)
        self.agents = tuple([i for i, _ in enumerate(start_goal)])
        self.best = None
        self.max_iter = max_iter
        self.iteration_counter = 0
        self.duplicate_cache = set()
        self.duplicate_counter = 0
        self.k_robustness = k_robustness
        self.discard_conflcicts_beyond = discard_conflicts_beyond
        spacetime_kwargs = {
            'wait_action_cost': wait_action_cost,
            'limit': limit,
        }
        self.cache = SpaceTimeAStarCache(self.g, [g for _, g in (self.start_goal)], kwargs=spacetime_kwargs)

    def update_state(self, state):
        self.cache.reset(state_only=True)
        assert len(state) == len(self.start_goal)
        self.start_goal = [(s, g) if s is not None else (g, g) for s, (_, g) in zip(state, self.start_goal)]
        for start, _ in self.start_goal:
            if (start is not None) and (start not in self.g.nodes()):
                raise nx.NodeNotFound()
        self.root = CBSNode(constraints=self.agent_constraints)
        self.best = None
        self.iteration_counter = 0
        self.duplicate_cache = set()
        self.duplicate_counter = 0

    def setup(self):
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
        found = False
        for _ in range(self.max_iter):
            if not self.step():
                found = True
                break
        if not found:
            logging.warning(f'max iter of {self.max_iter} is reached.')
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
    cost = {len(self.cache.cache)},
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
        if self.discard_conflcicts_beyond is not None:
            limit = self.discard_conflcicts_beyond
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
            sn, gn = self.start_goal[agent]
            path, cost = self.cache.get_path(sn, gn, node_constraints=nc)
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
        if self.discard_conflcicts_beyond is not None:
            limit = self.discard_conflcicts_beyond
        node.conflicts = compute_all_k_conflicts(node.solution, limit=limit, k=self.k_robustness)

        if not len(node.conflicts):
            logging.debug("set node to final because no conflicts")

            node.valid = True
            node.final = True
            self.update_best(node)
            return node
        return self.expand_node(node)

    def expand_node(self, node):
        node_expansion_heuristic = self.compute_node_conflict_heurstic(node)
        # get one of the conflicts, to create constraints from
        # each conflict must be resolved in some way, so we pick the most expensive conflict first
        # by picking the most expensive conflict, we generate less solutions with good fitness
        working_conflict = sorted(node.conflicts, key=lambda x: node_expansion_heuristic[x], reverse=True)[0]
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
        logging.info("no children added, node is not valid")
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


def check_nodes_connected(graph, paths):
    for p in paths:
        for e in zip(p[:-1], p[1:]):
            if e[0] == e[1]:
                continue
            if e not in graph.edges():
                return False
    return True


def prioritized_plans(graph, start_goal, constraints=frozenset(), limit=10, pad_paths=True, weight=None):
    solution = []
    if weight is None:
        weight = "dist"
    compute_normalized_weight(graph, weight)
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


class CDM_CR:
    def __init__(self,
                 g,
                 starts,
                 goals,
                 limit: int = 10,
                 wait_action_cost: float = 1.0001,
                 weight: str = None,
                 k_robustness: int = 1,
                 pad_paths: bool = False,
                 social_reward: float = 0.0,
                 anti_social_punishment: float = 0.0,
                 discard_conflicts_beyond=None):
        self.discard_conflicts_beyond = discard_conflicts_beyond
        assert len(starts) == len(goals)
        self.g = g
        compute_normalized_weight(self.g, weight)
        self.weight = "weight"
        self.starts = starts
        self.goals = goals
        self.agents = tuple(i for i, _ in enumerate(goals))
        self.social_reward = social_reward
        self.anti_social_punishment = anti_social_punishment
        self.g = g.to_directed()
        self.priority_map = self.g.copy()
        self.priorities = []
        self.priorities_in = []
        self.limit = limit
        self.k_robustness = k_robustness
        self.pad_paths = pad_paths
        astar_kwargs = {
            'limit': limit,
            'wait_action_cost': wait_action_cost
        }
        self.cache = SpaceTimeAStarCache(self.g, kwargs=astar_kwargs)
        self.constraints = []

    def run(self):
        conflicts = [None]
        # [None] is a placeholder to enter the loop initially
        while conflicts:
            # plan paths, based on current constraints
            solution = []
            costs = 0

            for agent, _ in enumerate(self.starts):
                start = self.starts[agent]
                goal = self.goals[agent]
                nc = frozenset([(c.node, c.time) for c in self.constraints if c.agent == agent])
                path, cost = self.cache.get_path(start, goal, frozenset(nc))
                if not path and start is not None:
                    logging.warning(f"no path found {start} - {goal}")
                    raise nx.NetworkXNoPath
                costs += cost
                solution.append(path)
                logging.info(f'start: {start}, goal: {goal}, path: {path}')

            # find conflicts
            conflicts = self.find_conflicts(solution)
            if not conflicts:
                return solution
            self.constraints = self.resolve_first_conflict(conflicts)
            # if this happens - the planning has no more nodes left to prioritize
            if self.constraints is None:
                return solution
            # go back to replanning (restart while)
        return solution

    def find_conflicts(self, solution):
        if not solution_valid(solution):
            logging.info(f'invalid solution: {solution}')
        limit = self.limit if self.pad_paths else None
        if self.discard_conflicts_beyond is not None:
            limit = self.discard_conflicts_beyond
        conflicts = compute_all_k_conflicts(solution, limit=limit, k=self.k_robustness)
        return conflicts

    def resolve_first_conflict(self, conflicts):
        logging.info(f"conflicts: {conflicts}")
        priorities = {}
        # compute priorities for decisions
        for conflict in conflicts:
            for x in conflict.conflicting_agents:
                if x.node not in priorities:
                    priorities[x.node] = 0
                priorities[x.node] += 1 / x.time
                if x.node in self.priorities:
                    priorities[x.node] = 0
        # priorities now holds all nodes with conflicts as key, agent_i * time_i as value
        logging.info(f'priorities: {priorities}')
        highest_priority_node = max(priorities, key=priorities.get)
        if priorities[highest_priority_node] == 0:
            return None
        logging.info(f'chosen: {highest_priority_node}')
        options = self.priority_map.in_edges(nbunch=highest_priority_node, data=True)
        options = copy.deepcopy(list(options))
        logging.info(f'options for priority edges: {options}')

        # compute decsion quality for the options, for the involved agents
        qualities = self.compute_qualities(options)
        logging.info(f'qualities: {qualities}')

        # make the decision
        decision = options[decision_function(qualities)]
        logging.info(f'decision: {decision}')
        self.implement_decision(decision)
        # create constraints from priority map
        return self.create_constraints_from_prio_map()

    def implement_decision(self, decision):
        # update the priority map with the descion
        # delete all edges going to the node $edge[1]
        edges = [e for e in self.priority_map.in_edges(decision[1])]
        for e in list(edges):
            self.priority_map.remove_edge(*e)
            self.g.edges[e[0], e[1]]["weight"] += self.anti_social_punishment

        # social reward only affects the global grap, we have to deduct the anti-social before
        self.g.edges[decision[0], decision[1]]["weight"] += -1 * self.social_reward - self.anti_social_punishment
        # insert $edge
        self.priority_map.add_edge(decision[0], decision[1], **decision[2])
        self.priority_map.edges[decision[0], decision[1]]["weight"] -= self.social_reward
        self.priorities.append(decision[1])
        self.priorities_in.append(decision[0])

        # recurse, to remove multiplel edges in narrow passages at once
        # outgoing edge
        if len(self.g.out_edges(nbunch=decision[1])) == 1:
            self.implement_decision(self.g.out_edges(nbunch=decision[1], data=True))
        if len(self.g.in_edges(nbunch=decision[0])) == 1:
            self.implement_decision(self.g.in_edges(nbunch=decision[0], data=True))

    def create_constraints_from_prio_map(self):
        # for each conflict, check which agent goes against prioritymap and update path accordingly
        constraints = self.constraints
        solution, costs = [], 0
        recompute_needed = True
        while recompute_needed:
            solution, costs = [], 0
            for agent, _ in enumerate(self.starts):
                start = self.starts[agent]
                goal = self.goals[agent]
                nc = frozenset([(c.node, c.time) for c in self.constraints if c.agent == agent])
                path, cost = self.cache.get_path(start, goal, frozenset(nc))
                if path is None:
                    logging.info("no path left, a solution is not possible")
                    raise nx.NetworkXNoPath()
                costs += cost
                solution.append(path)

            # find conflicts
            recompute_needed = False
            conflicts = self.find_conflicts(solution)
            for conflict in conflicts:
                # find out if conflict involves node with priorities
                for c in conflict.conflicting_agents:
                    logging.info(c)
                    # edge = t-1, t
                    edge = solution[c.agent][c.time - 1:c.time + 1]
                    logging.info(edge)
                    # this happens if the conflict occurs right at the first time step
                    if len(edge) < 2:
                        continue
                    if edge in self.priority_map.edges():
                        continue
                    constraints.append(c)
                    recompute_needed = True
                # recompute path for those agents, recompute conficts
            logging.info(f"constraints: {constraints}")
        return constraints

    def compute_qualities(self, options) -> list:
        """compute one quality for each option for each agent"""
        path_costs = [nx_shortest(self.priority_map, self.starts[i], self.goals[i], weight=self.weight)
                      for i, _ in enumerate(self.agents) if self.starts[i] is not None]
        qualities = []
        for o in options:
            qualities.append(self.evaluate_option(o, path_costs=path_costs))
        qualities = np.array(qualities)
        qualities = qualities - np.min(qualities, axis=0)
        return qualities

    def update_state(self, state):
        self.starts = state
        self.cache.reset(state_only=True)
        # We may need to update/fix the priority map.
        # currently we don't

        # however: we create constraints from the existing priority map, before starting the planning
        self.constraints = []
        self.create_constraints_from_prio_map()

    def evaluate_option(self, edge, path_costs=None):
        """evaluate giveing priority to a given edge (option)"""
        if path_costs is None:
            path_costs = [nx_shortest(self.priority_map, self.starts[i], self.goals[i], weight=self.weight)
                          for i, _ in enumerate(self.agents) if self.starts[i] is not None]

        # delete all edges going to the node $edge[1]
        edges = [e for e in self.priority_map.in_edges(edge[1])]
        # logging.info(f"edes: {edges}")
        for e in list(edges):
            self.priority_map.remove_edge(*e)

        # insert $edge
        self.priority_map.add_edge(edge[0], edge[1], **edge[2])

        # compute all paths and their cost
        # if no path is found, cost = np.inf
        new_path_costs = [nx_shortest(self.priority_map, self.starts[i], self.goals[i], weight=self.weight)
                          for i, _ in enumerate(self.agents) if self.starts[i] is not None]

        logging.info(f'edge: {edge[0:2]}, old cost: {path_costs}, new cost: {new_path_costs}')
        # compute the difference in path cost
        # np.inf - np.inf = np.nan
        cost_diff = np.array(path_costs) - np.array(new_path_costs)
        return cost_diff


if __name__ == "__main__":
    pass
