from asyncio.log import logger
from dataclasses import dataclass
from typing import Iterable, Protocol, Any
import numpy as np
from numpy.typing import ArrayLike
import networkx as nx
from networkx.algorithms.shortest_paths.weighted import _weight_function
from heapq import heappush, heappop
from itertools import count, zip_longest
from functools import partial
from typing import Union
import copy

from polygonal_roadmaps.environment import Environment
import logging

class Planner(Protocol):
    environment: Environment
    replan_required: bool
    history: list[list[int | None]]
    def __init__(self, environment: Environment, **kwargs: dict) -> None:
        """all plans should be initialized with an environment, general parameters and problem specific parameters passed as keyword arguments

        :param environment: environment
        :param planning_problem_parameters: environment parameters
        """
        ...
    
    def create_plan(self) -> list[list[int | None]]:
        ...


@dataclass(eq=True, frozen=True, init=True)
class NodeConstraint:
    agent: int
    time: int
    node: int


def pred_to_list(g, pred, start, goal) -> list:
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


def pad_path(path: list, limit: Union[int, None] = None) -> list:
    if limit is None:
        return path
    if path is None or not len(path):
        return []
    if limit > len(path):
        return path + [path[-1] for _ in range(limit - len(path))]
    if limit < len(path):
        return path[:limit]
    return path


def compute_node_conflicts(paths: list, limit: Union[int, None] = None) -> frozenset:
    node_occupancy = compute_node_occupancy(paths, limit=limit)

    conflicts = []
    for (t, node), agents in node_occupancy.items():
        if len(agents) > 1:
            conflicts.append(frozenset([NodeConstraint(time=t, node=node, agent=agent) for agent in agents]))
    return frozenset(conflicts)


def compute_node_occupancy(paths: list, limit: Union[int, None] = None) -> dict:
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

    def generate_constraints(self) -> set[frozenset]:
        """ generate a set of constraints for each conflicting agent, which resolves the conflict
        i.e., one agent can stay, all others have to move to resolve the conflict
        in a k=0 conflict, this means for each agent a set containing the other agents
        in a k!=0 conflict this means either the agent at time t moves or all agents at t-k move
        """
        if self.k == 0:
            return {self.conflicting_agents - {agent} for agent in self.conflicting_agents}
        t_agent = max(self.conflicting_agents, key=lambda x: x.time)
        return {frozenset(self.conflicting_agents - {t_agent}), frozenset({t_agent})}


def compute_k_robustness_conflicts(paths, limit: Union[int, None] = None, k: int = 0, node_occupancy: Union[dict, None] = None) -> set[Conflict]:
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
            if t - k > 0:
                constraints |= {NodeConstraint(time=t - k, node=node, agent=j) for j in conflicting_agents}
            conflicts.append(Conflict(k=k, conflicting_agents=frozenset(constraints)))
    return set(conflicts)

def compute_solution_robustness(solution, limit: Union[int, None] = None) -> int:
    maximum = max([len(path) for path in solution])
    for k in range(maximum):
        conflits = compute_all_k_conflicts(solution, limit=limit, k=k)
        if len(conflits):
            return k - 1
    return maximum


def compute_all_k_conflicts(solution, limit: Union[int, None] = None, k=1) -> frozenset[Conflict]:
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


def spatial_astar(G, source, target, weight=None) -> list:
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

def spatial_path_and_cost(G, source, target, weight):
    spatial_path = nx.shortest_path(G, source, target, weight)
    return spatial_path, sum_of_cost([spatial_path], G, weight)

def spacetime_astar_ccr(G, source, target, spacetime_heuristic=None, limit=100, wait_action_cost=.00001, belief=None, predecessors=None, node_contraints=None, preferred_nodes=None, inertia=0.2) -> tuple[list, float]:
                                       
    if belief is None:
        belief = {}

    if predecessors is None:
        predecessors = {}
    
    if node_contraints is None:
        node_contraints = set()
                                       
    def true_cost_heuristic(u):
        return nx.shortest_path_length(G, source=u, target=target, weight="weight")
    if spacetime_heuristic is None:
        spacetime_heuristic = true_cost_heuristic
    if preferred_nodes is None:
        preferred_nodes = {}

    def priority(n1, n2):
        # check the priority of edge n1 -> n2
        if n2 not in belief:
            return None
        return belief[n2].priorities[n1]
    
    weight_fn = _weight_function(G, "weight")
    c = count()
    nodes = {f'n{source}t0': {'n': source, 't': 0, 'cost': 0, 'parent': None}}
    queue = [(0, next(c), f'n{source}t0', 0, None)]                                   
    enqueued = {}
    explored = {}
    while queue:
        *_, curnode, dist, parent = heappop(queue)
        if nodes[curnode]['n'] == target:
            return spacetime_astar_calculate_return_path(nodes, explored, curnode, parent), dist
        if curnode in explored:
            if explored[curnode] is None:
                continue
            qcost, h = enqueued[curnode]
            if qcost < dist:
                continue
        explored[curnode] = parent
        
        # expand neighbours and add edges
        next_t = nodes[curnode]['t'] + 1
        if next_t > limit:
            continue
        node = nodes[curnode]['n']

        next_nodes = G[node].items()

        for next_node, w in next_nodes:
            # check if we are allowed
            # if not -- contiue this edge will not be used
            # k = 0 conflict:
                
            if (next_node, next_t) in node_contraints:
                continue
            if (next_node, next_t) in predecessors and next_node in belief:
                if priority(predecessors[next_node, next_t], next_node) >= priority(node, next_node):
                        continue
            if (next_node, next_t-1) in predecessors and next_node in belief:
                if priority(predecessors[next_node, next_t-1], next_node) >= priority(node, next_node):
                        continue
            if (next_node, next_t+1) in predecessors and next_node in belief:
                if priority(predecessors[next_node, next_t+1], next_node) >= priority(node, next_node):
                        continue
            
            ncost = dist + weight_fn(node, next_node, w)
            if next_node in preferred_nodes:
                ncost -= inertia
            if next_node == node:
                # add the wait action cost, waiting later is a tiny bit better
                ncost += wait_action_cost + 1 / next_t * 1e-4

            t_neighbour = f'n{next_node}t{next_t}'
            if t_neighbour in enqueued:
                qcost, h = enqueued[t_neighbour]
                if qcost < ncost:
                    # the path in the queue is better
                    continue
            else:
                h = spacetime_heuristic(next_node)
            nodes[t_neighbour] = {'n': next_node, 't': next_t}
            enqueued[t_neighbour] = ncost, h
            heappush(queue, (ncost + h, next(c), t_neighbour, ncost, curnode))
    raise nx.NetworkXNoPath(f"Node {target} not reachable from {source}")
        
def spacetime_astar(G, source, target, spacetime_heuristic=None, limit=100, node_constraints=None, wait_action_cost=1.0001) -> tuple[list, float]:
    
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
        if t > limit:
            continue
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
    cost = 0.0
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
    def __init__(self, constraints: frozenset[NodeConstraint] = frozenset()):
        self.children = ()
        self.fitness = np.inf
        self.valid = False
        self.paths = None
        self.conflicts : Union[None, frozenset] = None  # conflicts are found after plannig
        self.final = None
        self.solution = None
        self.constraints = constraints
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
            g.edges()[n1, n2]["weight"] = g.edges()[n1, n2][weight] / max_dist + np.random.rand() * 1e-6
    else:
        for n1, n2 in g.edges():
            g.edges()[n1, n2]['weight'] = 1.0001 + np.random.rand() * 1e-6


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

    def get_path(self, start, goal, node_constraints) -> tuple[list, float]:
        """try to look up a path from start to goal in the cache with specific contstraints
        if the path was searched before, we will return the path without new computation
        """
        def nc_tuples(node_constraints):
            nc = set()
            for c in node_constraints:
                if isinstance(c, tuple):
                    nc.add(c)
                elif isinstance(c, NodeConstraint):
                    nc.add((c.node, c.time))
                else:
                    raise ValueError(f"node_constraints must be a set of tuples or NodeConstraints, not {type(c)}")
            return frozenset(nc)
        node_constraints = nc_tuples(node_constraints) 
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


class CBS:
    def __init__(self,
                 env: Environment,
                 agent_constraints=None,
                 max_iter=10000,
                 repair_solutions=False,
                 ):
        self.start_goal = env.get_state_goal_tuples()
        self.env = env
        self.g = env.get_graph()
        for start, goal in self.start_goal:
            if start not in self.g.nodes() or goal not in self.g.nodes():
                raise nx.NodeNotFound()
        self.repair_solutions = repair_solutions
        compute_normalized_weight(self.g, self.env.planning_problem_parameters.weight_name)
        self.agent_constraints = frozenset() if agent_constraints is None else agent_constraints
        self.root = CBSNode(constraints=self.agent_constraints)
        self.agents = tuple([i for i, _ in enumerate(self.start_goal)])
        self.best = None
        self.max_iter = max_iter
        self.iteration_counter = 0
        self.duplicate_cache : set[NodeConstraint] = set()
        self.duplicate_counter = 0
        spacetime_kwargs = {
            'wait_action_cost': self.env.planning_problem_parameters.wait_action_cost,
            'limit': self.env.planning_horizon,
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
        if self.best is None and self.repair_solutions:
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
        limit = self.env.planning_problem_parameters.conflict_horizon if self.env.planning_problem_parameters.pad_path else None
        if self.env.planning_problem_parameters.conflict_horizon is not None:
            limit = self.env.planning_problem_parameters.conflict_horizon
        constraints = set(node.constraints)
        for agent in self.agents:
            solution, _ = self.compute_node_solution(CBSNode(constraints), max_agents=agent + 1)
            if not solution_valid(solution):
                # unsuccessful
                logging.debug("Repairing Solution is not successful")
                return
            for t, n in enumerate(pad_path(solution[agent], limit=limit)):
                # add constraints to the paths of all the following agents
                for k in range(self.env.planning_problem_parameters.k_robustness + 1):
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

    def compute_node_solution(self, node: CBSNode, max_agents=None) -> tuple[list, float]:
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

        limit = self.env.planning_problem_parameters.conflict_horizon if self.env.planning_problem_parameters.pad_path else None
        if self.env.planning_problem_parameters.conflict_horizon is not None:
            limit = self.env.planning_problem_parameters.conflict_horizon
        node.conflicts = compute_all_k_conflicts(node.solution, limit=limit, k=self.env.planning_problem_parameters.k_robustness)

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
                logging.warning(f"edge {e} not in graph, for path {p}")
                return False
    return True


def prioritized_plans(env: Environment,
                      constraints=frozenset()):
    planning_problem_parameters = env.planning_problem_parameters
    spacetime_kwargs = {
        'wait_action_cost': planning_problem_parameters.wait_action_cost,
        'limit': env.planning_horizon,
    }
    cache = SpaceTimeAStarCache(env.get_graph(), kwargs=spacetime_kwargs)
    solution : list[list[int]] = []
    compute_normalized_weight(env.get_graph(), planning_problem_parameters.weight_name)
    pad_limit : Union[int, None] = env.planning_horizon
    if not planning_problem_parameters.pad_path:
        pad_limit = None
    if planning_problem_parameters.conflict_horizon is not None:
        pad_limit = planning_problem_parameters.conflict_horizon
    for start, goal in env.get_state_goal_tuples():
        node_occupancy = compute_node_occupancy(solution, limit=pad_limit)
        constraints = set()
        for t, node in node_occupancy.keys():
            if t > 0:
                if planning_problem_parameters.conflict_horizon is not None and t > planning_problem_parameters.conflict_horizon:
                    continue
                constraints.add((node, t))
                constraints.add((node, t + 1))
        logging.debug(constraints)
        path, _ = cache.get_path(start, goal, node_constraints=frozenset(constraints))
        if path is None:
            raise nx.NetworkXNoPath
        solution.append(path)
    return solution


class CCRPlanner(Planner):
    def __init__(self, environment: Environment, social_reward=0.0, anti_social_punishment=0.0) -> None:
        # initialize the planner.
        # if the horizon is not None, we want to replan after execution of one step
        self.environment = environment
        self.replan_required = self.environment.planning_problem_parameters.conflict_horizon is None
        self.history = []

        self.g = self.environment.get_graph().to_directed()
        compute_normalized_weight(self.g, self.environment.planning_problem_parameters.weight_name)
        self.weight = "weight"
        self.agents = tuple(i for i, _ in enumerate(self.environment.goal))
        self.social_reward = social_reward
        self.anti_social_punishment = anti_social_punishment
        self.priority_map = self.g.copy()
        self.priorities:list[int|str] = []
        self.priorities_in:list[int|str] = []
        astar_kwargs = {
            'limit': self.environment.planning_horizon,
            'wait_action_cost': self.environment.planning_problem_parameters.wait_action_cost,
        }
        self.cache = SpaceTimeAStarCache(self.g, kwargs=astar_kwargs)
        self.constraints : list[NodeConstraint] = []

    def create_plan(self, *_):
        if self.replan_required:
            self.update_state(self.environment.state)
        plans = self.run()
        # reintroduce plan for those states that have already finished -> i.e., where state is None
        self.history.append({"solution": plans, "priorities": list(zip(self.priorities_in, self.priorities))})
        logging.info(f'plans: {plans}')
        ret = []
        for i, s in enumerate(self.environment.state):
            if s is not None:
                ret.append(plans[i] + [None])
            else:
                ret.append([None])
        ret = zip_longest(*ret, fillvalue=None)
        return list(ret)

    def run(self):
        conflicts = [None]
        # [None] is a placeholder to enter the loop initially
        while conflicts:
            # plan paths, based on current constraints
            solution = []
            costs = 0

            for agent, _ in enumerate(self.environment.state):
                start = self.environment.state[agent]
                goal = self.environment.goal[agent]
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
        limit = self.environment.planning_horizon if self.environment.planning_problem_parameters.pad_path else None
        if self.environment.planning_problem_parameters.conflict_horizon is not None:
            limit = self.environment.planning_problem_parameters.conflict_horizon
        conflicts = compute_all_k_conflicts(solution, limit=limit, k=self.environment.planning_problem_parameters.k_robustness)
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
        decision = decision_function(qualities)
        logging.info(f'decision: {decision}')
        self.implement_decision(options[decision])
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
            for agent, _ in enumerate(self.environment.state):
                start = self.environment.state[agent]
                goal = self.environment.goal[agent]
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
            for conflict in sorted(conflicts, key=lambda x: iter(x.conflicting_agents).__next__().time):
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

    def compute_qualities(self, options) -> ArrayLike:
        """compute one quality for each option for each agent"""
        path_costs = [nx_shortest(self.priority_map, self.environment.state[i], self.environment.goal[i], weight=self.weight)
                      for i, _ in enumerate(self.agents) if self.environment.state[i] is not None]
        q = []
        for o in options:
            q.append(self.evaluate_option(o, path_costs=path_costs))
        qualities = np.array(q)
        qualities = qualities - np.min(qualities, axis=0)
        return qualities

    def update_state(self, state):
        self.cache.reset(state_only=True)
        # We may need to update/fix the priority map.
        # currently we don't

        # however: we create constraints from the existing priority map, before starting the planning
        self.constraints = []
        self.create_constraints_from_prio_map()

    def evaluate_option(self, edge, path_costs=None):
        """evaluate giveing priority to a given edge (option)"""
        if path_costs is None:
            path_costs = [nx_shortest(self.priority_map, self.environment.state[i], self.environment.goal[i], weight=self.weight)
                          for i, _ in enumerate(self.agents) if self.environment.state[i] is not None]

        # delete all edges going to the node $edge[1]
        edges = [e for e in self.priority_map.in_edges(edge[1])]
        # logging.info(f"edes: {edges}")
        for e in list(edges):
            self.priority_map.remove_edge(*e)

        # insert $edge
        self.priority_map.add_edge(edge[0], edge[1], **edge[2])

        # compute all paths and their cost
        # if no path is found, cost = np.inf
        new_path_costs = [nx_shortest(self.priority_map, self.environment.state[i], self.environment.goal[i], weight=self.weight)
                          for i, _ in enumerate(self.agents) if self.environment.state[i] is not None]

        logging.info(f'edge: {edge[0:2]}, old cost: {path_costs}, new cost: {new_path_costs}')
        # compute the difference in path cost
        # np.inf - np.inf = np.nan
        cost_diff = np.array(path_costs) - np.array(new_path_costs)
        return cost_diff


class FixedPlanner(Planner):
    def __init__(self, environment: Environment, **kwargs) -> None:
        """ Planner used for testing purposes, a precomputed plan is passed as keyword argument

        :param environment: _description_
        :keyword plan: plann passed as keyword argument
        """
        planning_problem_parameters = environment.planning_problem_parameters
        self.environment = environment
        self.planning_problem_parameters = planning_problem_parameters
        self.replan_required = False
        self._plan = kwargs['plan']
        self.history = []

    def create_plan(self, *_):
        return self._plan


class PrioritizedPlanner(Planner):
    def __init__(self, environment: Environment, **kwargs) -> None:
        planning_problem_parameters = environment.planning_problem_parameters
        self.planning_problem_parameters = planning_problem_parameters
        self.replan_required = self.planning_problem_parameters.conflict_horizon is not None
        self.environment = environment
        self.kwargs = kwargs
        self.history = []

    def create_plan(self, *_):
        plans = prioritized_plans(self.environment, self.planning_problem_parameters, **self.kwargs)
        j = 0
        ret = []
        logging.info(f"state: {self.environment.state}")
        for i, s in enumerate(self.environment.state):
            if s is not None:
                ret.append(plans[j] + [None])
                j += 1
            else:
                ret.append([None])
        ret = zip_longest(*ret, fillvalue=None)
        self.history.append({"solution": plans})
        return list(ret)


class CBSPlanner(Planner):
    def __init__(self, environment: Environment, **kwargs) -> None:
        planning_problem_parameters = environment.planning_problem_parameters
        # initialize the planner.
        # if the horizon is not None, we want to replan after execution of one step
        self.replan_required = planning_problem_parameters.conflict_horizon is not None
        self.kwargs = kwargs
        self.environment = environment
        self.cbs = CBS(self.environment, **self.kwargs)
        self.history = []

    def create_plan(self, *_):
        if self.replan_required:
            self.cbs.update_state(self.environment.state)
        self.cbs.run()
        plans = list(self.cbs.best.solution)
        ret = []
        logging.info(f"state: {self.environment.state}")
        for i, s in enumerate(self.environment.state):
            if s is not None:
                ret.append(plans[i] + [None])
            else:
                ret.append([None])
        ret = zip_longest(*ret, fillvalue=None)
        self.history.append({"solution": plans})
        return list(ret)


class BeliefState:
    """Belief state of an agent"""

    def __init__(self, state, priorities):
        self.state = state
        self.priorities = priorities
        self.priorities[state] = np.inf

    def __add__(self, other):
        # if we add a scalar number, we add it to all priorities
        if isinstance(other, (int, float)):
            return BeliefState(self.state, {k: v + other for k, v in self.priorities.items()})
        # if we add another belief state, we add the priorities
        if isinstance(other, BeliefState):
            bs = BeliefState(self.state, {})
            for k in self.priorities.keys() | other.priorities.keys():
                v1 = 0 if k not in self.priorities else self.priorities[k]
                v2 = 0 if k not in other.priorities else other.priorities[k]
                bs.priorities[k] = v1 + v2
            return bs
        raise TypeError(f"unsupported operand type(s) for +: '{type(self)}' and '{type(other)}'")
    
    def __mul__(self, other):
        # if we multiply a scalar number, we multiply it to all priorities
        if isinstance(other, (int, float)):
            return BeliefState(self.state, {k: v * other for k, v in self.priorities.items()})
        raise TypeError(f"unsupported operand type(s) for +: '{type(self)}' and '{type(other)}'")
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __str__(self):
        return f"BeliefState(state={self.state}, priorities={self.priorities})"
    
    def __repr__(self):
        return self.__str__()
    
    def normalize(self):
        s = sum(v for v in self.priorities.values() if v != np.inf)
        self.priorities = {k: v / s for k, v in self.priorities.items()}

class CCRAgent:
    def __init__(self, graph: nx.Graph, state:int, goal:int, planning_problem_parameters, index: int, limit=100, inertia:float=0.2, block_steps=3):
        self.g = graph.copy()
        for node in self.g.nodes():
            self.g.add_edge(node, node, weight=planning_problem_parameters.wait_action_cost)
        self.state = state
        self.goal = goal
        self.planning_problem_parameters = planning_problem_parameters
        self.belief: dict[int, BeliefState] = {}
        self.other_paths: dict[int, list] = {}
        self.index = index
        self.plan: list[int] = []
        self.cost = np.inf
        self.conflicts = frozenset()
        self.limit = limit
        self.belief_changed = True
        self.goal_changed = True
        self.state_changed = True
        self.plans_changed = True
        self.inertia = inertia
        self.block_steps = block_steps
        self.compute_plan()
        
    def update_other_paths(self, other_paths):
        ret = self._update_other_paths(other_paths)
        if ret:
            self.get_conflicts()
        self.plans_changed |= ret
        return ret

    def _update_other_paths(self, other_paths):
        # remove own path
        op = {k: v for k, v in other_paths.items() if k != self.index}
        
        op = self.other_paths | op

        if op == self.other_paths:
            return False
        self.other_paths = op
        return True
        
    def update_state(self, state):
        if state == self.state:
            return
        self.state_changed = True
        self.state = state
        self.compute_plan()
        
    def update_goal(self, goal):
        if goal == self.goal:
            return
        self.goal_changed = True
        self.goal = goal
        self.compute_plan()

    def get_path(self, source, goal):
        pred = {}
        preferred_nodes = set(self.plan)
        for path in self.other_paths.values():
            for i, node in enumerate(path[1:]):
                if self.planning_problem_parameters.conflict_horizon and i > self.planning_problem_parameters.conflict_horizon:
                    continue
                # we need to check, that there is no other edge with more priority used for this node
                if (node, i+1) in pred:
                    # get priority of this and the new one
                    if node not in self.belief:
                        continue
                    if pred[node, i+1] > self.belief[node].priorities[path[i]]:
                        continue
                pred[node, i+1] = path[i]
        # we are not allowed to go to the position of another robot at and t=1, because this will be a conflict that is not possible to be resolved
        nc = set()
        for t in range(1, 1+self.block_steps):
            nc |= set((p[0], t) for p in self.other_paths.values())
        try:
            return spacetime_astar_ccr(self.g, source, goal, limit=self.limit, belief=self.belief, predecessors=pred, node_contraints=nc, preferred_nodes=preferred_nodes, inertia=self.inertia)
        except nx.NetworkXNoPath:
            return [self.state], np.inf


    def get_plan(self) -> list[int]:
        self.compute_plan()
        return self.plan
    
    def _get_conflicts(self, path):
        conflicts = compute_all_k_conflicts([self.plan, path], limit=self.planning_problem_parameters.conflict_horizon, k=self.planning_problem_parameters.k_robustness)
        self.conflicts = frozenset().union(conflicts, self.conflicts)

    def get_conflicts(self):
        self.conflicts = frozenset()
        for _, path in self.other_paths.items():
            self._get_conflicts(path)
        return self.conflicts
    
    def replan_needed(self):
        if self.state is None:
            return False
        if self.goal is None:
            return False
        if self.plan is None:
            return True
        if self.belief_changed:
            return True
        if self.state_changed:
            return True
        if self.goal_changed:
            return True
        if self.plans_changed:
            return True
        return False
    
    def compute_plan(self):
        if not self.replan_needed():
            return
        self.plan, self.cost = self.get_path(self.state, self.goal)
        self.belief_changed = False
        self.state_changed = False
        self.goal_changed = False
        self.plans_changed = False
        
    def make_plan_consistent(self):
        """Update the plan of an agent
        When other agents plans are passed, the agent will respect their right of way according to its belief


        :param other_plans: list of plans of other agents
        """
        plan = self.plan
        self.compute_plan()
        self.get_conflicts()
        return plan != self.plan

    def get_cdm_node(self, re_decide_belief=False) -> set[Any]:
        if not len(self.get_conflicts()):
            return set()
        # just return the nodes with a conflict
        nodes = set()
        for c in self.get_conflicts():
            for ca in c.conflicting_agents:
                if ca.node in self.belief and not re_decide_belief:
                    continue
                nodes.add(ca.node)
        # randomly pick one of the nodes
        return nodes
    
    def get_cdm_opinion(self, node) -> BeliefState:
        """Generate a tuple of qualities for each edge going to the node

        :param node: the node in the graph which we make a decision for
        :return: tuple of qualities for each edge going to the node
        """
        # get all edges going to the node
        bs = self.compute_qualities(node) 
        if node not in self.belief:
            return bs
        return 0.5 * bs + 0.5 * self.belief[node]

    def compute_qualities(self, node):
        # compute qualities using flow graph
        if self.state is None or self.goal is None or self.state == self.goal:
            return BeliefState(node, {e[0]: 0.0 for e in self.g.in_edges(node)})
        options = [e for e, _ in self.g.in_edges(node)]
        fg = self.compute_flow_graph()
        qualities = self.compute_qualities_flow(fg, node, options)
        bs = BeliefState(state=node, priorities={e: q for e, q in qualities})
        #bs.normalize()
        # logging.warn(f"bs: {bs}")
        return bs

    def compute_flow_graph(self):
        fg = self.g.copy()
        # add flow weight and capacity
        for e in fg.edges():
            fg.edges[e]['flow_weight'] = int(self.g.edges[e]['weight'] * 100)
            fg.edges[e]['capacity'] = 50
            
        for node, bel in self.belief.items():
            # sort priorities
            bel_sorted = sorted(bel.priorities.items(), key=lambda x: x[1])
            # the capacity of the best edge should be 50
            if len(bel_sorted) == 0:
                continue
            fg.edges[bel_sorted[0][0], bel.state]["capacity"] = 100
            fg.edges[bel_sorted[0][0], bel.state]["flow_weight"] = 0
            # the capacity of the other edges should be 4, 3,...0
            for i, n in enumerate(bel_sorted[1:]):
                fg.edges[n[0], node]["capacity"] = len(bel_sorted) - i
        return fg

    def compute_quality(self, _, node:int, neighbour:int)->float:
        # now we compute the path with only one of those edges leading to the node present in the graph
        # low random number, if edge in path
        # high random number, if edge not in path
        q = 0.001 * (np.random.rand() - 0.5)
        if node == neighbour:
            return q 
        # if neighbour -> node is in the plan: add 1
        for i, n in enumerate(self.plan[:-1]):
            if n == neighbour:
                q += 0.1
                if self.plan[i+1] == node:
                    q += 1.0
        # if node -> neighbour is in the plan: add -1
        for i, n in enumerate(self.plan[:-1]):
            if n == node:
                q += 0.1
                if self.plan[i+1] == neighbour:
                    q -= 1.0
        return q
    
    def compute_loss(self, g, node:int, neighbour:int)->float:
        p = []
        if self.state == self.goal:
            return 0.0
        if self.state is None:
            return 0.0
        if self.goal is None:
            return 0.0
        try:
            p = nx.shortest_path(g, source=self.state, target=self.goal, weight=self.planning_problem_parameters.weight_name)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return 0.0
        if node not in p:
    

            return 0.0 
        if neighbour in p:
            if p.index(neighbour) == p.index(node) - 1:
                return 0.0 
        
        q0 = nx.shortest_path_length(g, source=self.state, target=self.goal, weight=self.planning_problem_parameters.weight_name)
        g = g.copy()
        for e in g.in_edges(node):
            g.edges[e[0], e[1]]["weight"] = 10e10
        g.edges[neighbour, node]["weight"] = self.g.edges[neighbour, node]["weight"]
        # compute the path
        try:
            q = nx.shortest_path_length(g, source=self.state, target=self.goal, weight=self.planning_problem_parameters.weight_name)
            return q - q0
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return 10e10
        return 0.0
    
    def compute_qualities_flow(self, g, node, options):
        # compute the min cost flow
        flow = nx.max_flow_min_cost(g, self.state, self.goal, weight="flow_weight", capacity="capacity")
        # the flow indicates the quality of the edge
        # compute the maximum flow for normalization
        max_flow = sum(flow[self.state].values())

        qualities = [(e, flow[e][node] / max_flow +np.random.rand()*0.1) for e in options]
        qualities.append((node, np.inf))
        return qualities

    def set_belief(self, node, belief):
        self.belief_changed = True
        self.belief[node] = belief
        
    def delete_belief(self, node):
        if node not in self.belief:
            return
        self.belief_changed = True
        del self.belief[node]


class PriorityAgent:
    def __init__(self, graph: nx.Graph, state:int, goal:int, planning_problem_parameters, index: int, limit=100, inertia:float=0.2, block_steps=3, priorities=None):
        self.g = graph.copy()
        for node in self.g.nodes():
            self.g.add_edge(node, node, weight=planning_problem_parameters.wait_action_cost)
        self.state = state
        self.goal = goal
        self.planning_problem_parameters = planning_problem_parameters
        self.other_paths: dict[int, list] = {}
        if priorities is None:
            self.priorities = {index: 0}
        else:
            self.priorities = priorities
        
        self.index = index
        self.plan: list[int] = []
        self.cost = np.inf
        self.conflicts = frozenset()
        self.limit = limit
        self.goal_changed = True
        self.state_changed = True
        self.plans_changed = True
        self.inertia = inertia
        self.block_steps = block_steps
        self.compute_plan()
        
    def update_other_paths(self, other_paths):
        ret = self._update_other_paths(other_paths)
        if ret:
            self.get_conflicts()
        self.plans_changed |= ret
        return ret

    def _update_other_paths(self, other_paths):
        # remove own path
        op = {k: v for k, v in other_paths.items() if k != self.index}
        
        op = self.other_paths | op

        if op == self.other_paths:
            return False
        self.other_paths = op
        return True
        
    def update_state(self, state):
        if state == self.state:
            return
        self.state_changed = True
        self.state = state
        self.compute_plan()
        
    def update_goal(self, goal):
        if goal == self.goal:
            return
        self.goal_changed = True
        self.goal = goal
        self.compute_plan()

    def get_path(self, source, goal):
        preferred_nodes = set(self.plan)
        nc = set()
        for agent, path in self.other_paths.items():
            if agent not in self.priorities:
                self.priorities[agent] = 0
            if self.priorities[agent] < self.priorities[self.index]:
                continue
            horizon = self.planning_problem_parameters.conflict_horizon
            if horizon is None:
                horizon = len(path)
            for i, node in enumerate(path[1:horizon+2]):
                # we need to check, that there is no other edge with more priority used for this node
                nc.add((node, i))
                nc.add((node, i+1))
                nc.add((node, i+2))
                    
        # we are not allowed to go to the position of another robot at and t=1, because this will be a conflict that is not possible to be resolved
        for t in range(1, 1+self.block_steps):
            nc |= set((p[0], t) for p in self.other_paths.values())
        try:
            return spacetime_astar_ccr(self.g, source, goal, limit=self.limit, belief=None, predecessors=set(), node_contraints=nc, preferred_nodes=preferred_nodes, inertia=self.inertia)
        except nx.NetworkXNoPath:
            return [self.state], np.inf


    def get_plan(self) -> list[int]:
        self.compute_plan()
        return self.plan
    
    def _get_conflicts(self, path):
        conflicts = compute_all_k_conflicts([self.plan, path], limit=self.planning_problem_parameters.conflict_horizon, k=self.planning_problem_parameters.k_robustness)
        self.conflicts = frozenset().union(conflicts, self.conflicts)

    def get_conflicts(self):
        self.conflicts = frozenset()
        for _, path in self.other_paths.items():
            self._get_conflicts(path)
        return self.conflicts
    
    def replan_needed(self):
        if self.state is None:
            return False
        if self.goal is None:
            return False
        if self.plan is None:
            return True
        if self.state_changed:
            return True
        if self.goal_changed:
            return True
        if self.plans_changed:
            return True
        return False
    
    def compute_plan(self):
        if not self.replan_needed():
            return
        self.plan, self.cost = self.get_path(self.state, self.goal)
        self.state_changed = False
        self.goal_changed = False
        self.plans_changed = False
        
    def make_plan_consistent(self):
        """Update the plan of an agent
        When other agents plans are passed, the agent will respect their right of way according to its belief


        :param other_plans: list of plans of other agents
        """
        plan = self.plan
        self.compute_plan()
        return plan != self.plan

class PriorityAgentPlanner(Planner):
    def __init__(self, environment: Environment, priority_method=None, max_iter=100, **kwargs: dict) -> None:
        self.max_iter = max_iter
        self.environment = environment
        self.replan_required = self.environment.planning_problem_parameters.conflict_horizon is None
        self.history = []
        self.g = self.environment.get_graph().to_directed()
        compute_normalized_weight(self.g, self.environment.planning_problem_parameters.weight_name)
        self.weight = "weight"
        priorities = None
        if priority_method is not None:
            if priority_method == "index":
                priorities = {i: i for i, _ in enumerate(self.environment.get_state_goal_tuples())}
            elif priority_method == "random":
                priorities = {i: np.random.rand() for i, _ in enumerate(self.environment.get_state_goal_tuples())}
            elif priority_method == "same":
                priorities = {i: 0 for i, _ in enumerate(self.environment.get_state_goal_tuples())}
        
        self.agents = [
            PriorityAgent(self.g, sg[0], sg[1], self.environment.planning_problem_parameters, i, priorities=priorities)
            for i, sg in enumerate(self.environment.get_state_goal_tuples())
        ]
    
    def create_plan(self, *_) -> list[list[int | None]]:
        self.update_state(self.environment.state)
        self.planning_loop()
        # post-process plans to right format:
        plans = [agent.get_plan() for agent in self.agents]
        plans = [p + [None] for p in plans]
        ret = zip_longest(*plans, fillvalue=None)
        self.history.append({"solution": plans})
        return list(ret)

    def update_state(self, state):
        for i, a in enumerate(self.agents):
            a.update_state(state[i])
        
    def planning_loop(self):
        for _ in range(self.max_iter):
            self.make_all_plans_consistent()
            self.update_all_paths()

            conflicts = [bool(len(a.get_conflicts())) for a in self.agents]
            if not any(conflicts):
                return
    
    def update_all_paths(self):
        self.plans = {a.index: a.get_plan() for a in self.agents}
        for a in self.agents:
            a.update_other_paths(self.plans)
    
    def make_all_plans_consistent(self):
        self.update_all_paths()
        for a in self.agents:
            # this will change plans
            a.make_plan_consistent()
            self.update_all_paths()

class CCRv2(Planner):
    def __init__(self, environment: Environment, max_iter=100, **kwargs) -> None:
        # initialize the planner.
        # if the horizon is not None, we want to replan after execution of one step
        self.max_iter = max_iter
        self.environment = environment
        self.replan_required = self.environment.planning_problem_parameters.conflict_horizon is None
        self.history = []

        self.g = self.environment.get_graph().to_directed()
        compute_normalized_weight(self.g, self.environment.planning_problem_parameters.weight_name)
        self.weight = "weight"
        
        self.agents = [
            CCRAgent(self.g, sg[0], sg[1], self.environment.planning_problem_parameters, i)
                for i, sg in enumerate(self.environment.get_state_goal_tuples())
                ]
        
    def make_cdm_decision(self):
        node = self.pick_cdm_node()
        if node is None:
            return False
        decision = self.agents[0].get_cdm_opinion(node)
        for a in self.agents[1:]:
            decision += a.get_cdm_opinion(node)
        for a in self.agents:
            a.set_belief(node, decision)
        return True
    
    def pick_cdm_node(self):
        # get random node for cdm
        nodes = set()
        for a in self.agents:
            nodes |= a.get_cdm_node()
        if not len(nodes):
            return None
        return np.random.choice(list(nodes))
    
    def planning_loop(self):
        for _ in range(self.max_iter):
            # update other paths
            # make plans consistent
            self.update_all_paths()
            self.make_all_plans_consistent()
            self.update_all_paths()

            # make cdm decision, once all plans are consistent with the belief

            conflicts = [bool(len(a.get_conflicts())) for a in self.agents]
            if not any(conflicts):
                return
            #if not self.make_cdm_decision():
            #    return
            self.make_cdm_decision()
            
        raise nx.NetworkXNoPath()

    def update_all_paths(self):
        self.plans = {a.index: a.get_plan() for a in self.agents}
        for a in self.agents:
            a.update_other_paths(self.plans)

    def make_all_plans_consistent(self):
        for a in self.agents:
            # this will change plans
            a.make_plan_consistent()
            
    def update_state(self, state):
        for i, a in enumerate(self.agents):
            a.update_state(state[i])
                
    def create_plan(self, *_) -> list[list[int | None]]:
        """Create a plan for all agents
        
        :return: list of plans for all agents
        """
        self.update_state(self.environment.state)
        self.planning_loop()
        # post-process plans to right format:
        plans = [agent.get_plan() + [None] for agent in self.agents]
        ret = zip_longest(*plans, fillvalue=None)
        self.history.append({"solution": plans})
        return list(ret)


if __name__ == "__main__":
    pass

