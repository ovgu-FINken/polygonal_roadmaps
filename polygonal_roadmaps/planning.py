from asyncio.log import logger
from dataclasses import dataclass
from typing import Callable, Iterable, Protocol, Any
import numpy as np
from numpy.typing import ArrayLike
import networkx as nx
from networkx.algorithms.shortest_paths.weighted import _weight_function
from heapq import heappush, heappop
from itertools import count, zip_longest
from functools import partial
from typing import Union
from abc import ABC, abstractmethod
from functools import lru_cache
import copy
import random

from polygonal_roadmaps.environment import Environment, PlanningProblemParameters
import logging


@dataclass(eq=True, init=True)
class Plans():
    plans: list[list[int]]
    
    def __iter__(self):
        return iter(self.plans)
    
    def __getitem__(self, key):
        return self.plans[key]
    
    def __str__(self):
        s = "Plans:\n--" + "\n--".join([str(p) for p in self.plans])
        return s
    
    def as_state_list(self):
        return list(zip_longest(*self.plans, fillvalue=None))
    
    @classmethod
    def from_state_list(cls, state_list):
        if not len(state_list):
            return cls([])
        plans = [[] for _ in state_list[0]]
        for _, state in enumerate(state_list):
            for i, s in enumerate(state):
                if s is None:
                    continue
                plans[i].append(s)
            
        return cls(plans)

    def contains_conflicts(self, env:Environment|None=None, limit: Union[int, None] = None, k: int = 1) -> bool:
        if env is not None:
            k = env.planning_problem_parameters.k_robustness
            limit = env.planning_problem_parameters.conflict_horizon
        return len(compute_all_k_conflicts(self, limit=limit, k=k)) > 0
    
    def transitions_are_valid(self, env) -> bool:
        for _, path in enumerate(self):
            for n1, n2 in zip(path[:-1], path[1:]):
                if n1 is None or n2 is None:
                    continue
                if n1 == n2:
                    continue
                if not env.g.has_edge(n1, n2):
                    logging.warn(f'plan contains edge between {n1} and {n2}, which is not part of the graph')
                    return False
        return True
    
    def end_in_goals(self, env: Environment) -> bool:
        for i, plan in enumerate(self.plans):
            if env.state[i] is None:
                continue
            if plan[-1] != env.goal[i]:
                logging.warn(f"plan {i}: {plan} does not end in goal {env.goal[i]}")
                return False
        return True
    
    def start_in_start(self, env: Environment) -> bool:
        for i, plan in enumerate(self.plans):
            if env.state[i] is None:
                continue
            if plan[0] != env.state[i]:
                logging.warn(f"plan {i}: {plan} does not start in start {env.state[i]}")
                return False
        return True

    def is_valid(self, env: Environment|None = None, k:int=1, limit:int=None):
        """ check if the plan is valid
            - check if there are conflicts within the plan
            - check if the plan is valid with in the given environment
            - if no environment is passed, only the conflicts are checked
            - if an environmnet is passed, k and limit are overwritten by the environment's parameters
        """
        # check if there are conflicts within the plan:
        if self.contains_conflicts(env, limit=limit, k=k):
            logging.warn("plan contains conflicts")
            return False
        
        # check if all edges are valid within the environment
        if env is None:
            return True
        
        return self.transitions_are_valid(env) and self.end_in_goals(env) and self.start_in_start(env)
    
    def get_state(self, t):
        sl = self.as_state_list()
        if not len(sl):
            return ()
        if t >= len(sl):
            return (None for _ in sl[0])
        return sl[t]

    def get_next_state(self):
        return self.get_state(1)


class Planner(ABC):
    environment: Environment
    replan_required: bool
    history: list[list[int | None]]

    @abstractmethod
    def __init__(self, environment: Environment, **kwargs: dict) -> None:
        """all plans should be initialized with an environment, general parameters and problem specific parameters passed as keyword arguments

        :param environment: environment
        :param planning_problem_parameters: environment parameters
        """
        pass

    @abstractmethod
    def create_plan(self) -> Plans:
        pass


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
            if node is None:
                continue
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

def compute_k_robustness_conflicts_new(paths, limit: Union[int, None] = None, k: int = 0, node_occupancy: Union[dict, None] = None) -> set[Conflict]:
    def compute_agent_occupancy(paths, limit: Union[int, None] = None) -> dict:
        agent_occupancy = {}
        for i, path in enumerate(paths):
            agent_occupancy[i] = { (t,  node) for t, node in enumerate(pad_path(path, limit=limit)) }
        return agent_occupancy
    shifted_paths = [p[k:] for p in paths]
    occupancy = compute_agent_occupancy(paths, limit=limit)
    shifted_occupancy = compute_agent_occupancy(shifted_paths, limit=limit)
    conflicts = set()
    for i, _ in enumerate(paths):
        for j, _ in enumerate(paths):
            if i == j:
                continue
            intersetion = occupancy[i] & shifted_occupancy[j]
            # when the intersection is not empty, there are conflicts
            if len(intersetion):
                conflicts.add(Conflict(k=k, conflicting_agents=frozenset(NodeConstraint(time=occupancy[i][0], node=occupancy[i][1], agent=i), NodeConstraint(time=shifted_occupancy[j][0]+2, node=shifted_occupancy[j][1], agent=j))))
                              
    return conflicts

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


def temporal_node_list(G, limit, node_constraints):
    nodes = G.nodes()
    nodelist = []
    for t in range(limit):
        nodelist += [(f'n{n}t{t}', {'n': n, 't': t}) for n in nodes if (n, t) not in node_constraints]
    return nodelist

def spatial_path_and_cost(G, source, target, weight):
    spatial_path = nx.shortest_path(G, source, target, weight)
    return spatial_path, sum_of_cost([spatial_path], G, weight)


@lru_cache(maxsize=None)
def cached_shortest_path(G, source, target, weight):
    return nx.shortest_path(G, source, target, weight)

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
        self.env.state = state
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
            if e[0] is None or e[1] is None:
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
    for start, goal in zip(env.state, env.goal):
        if start is None:
            solution.append([])
            continue
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
        return Plans(plans)

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
        return Plans.from_state_list(self._plan)


class PrioritizedPlanner(Planner):
    def __init__(self, environment: Environment, **kwargs) -> None:
        planning_problem_parameters = environment.planning_problem_parameters
        self.planning_problem_parameters = planning_problem_parameters
        self.replan_required = self.planning_problem_parameters.conflict_horizon is not None
        self.environment = environment
        self.kwargs = kwargs
        self.history = []

    def create_plan(self, env: Environment, *_):
        self.environment = env
        plans = prioritized_plans(self.environment, self.planning_problem_parameters, **self.kwargs)
        return Plans(plans)


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

    def create_plan(self, env: Environment, *_):
        self.cbs.update_state(env.state)
        self.cbs.run()
        return Plans(self.cbs.best.solution)


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
    def __init__(self, graph: nx.Graph, state:int, goal:int, planning_problem_parameters, index: int, limit=100, inertia:float=0.2, block_steps=3, quality_metric=None):
        self.g = graph.copy()
        for node in self.g.nodes():
            self.g.add_edge(node, node, weight=planning_problem_parameters.wait_action_cost)
        self.state = state
        self.goal = goal
        self.planning_problem_parameters = planning_problem_parameters
        self.quality_metric = quality_metric
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
        # self.compute_plan()
        
    def update_goal(self, goal):
        if goal == self.goal:
            return
        self.goal_changed = True
        self.goal = goal
        # self.compute_plan()

    def get_path(self, source, goal) -> tuple[list, float]:
        if source is None:
            return [None], np.inf
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
                    if self.belief[node].priorities[pred[node, i+1]] > self.belief[node].priorities[path[i]]:
                        continue
                pred[node, i+1] = path[i]
        # we are not allowed to go to the position of another robot at and t=1, because this will be a conflict that is not possible to be resolved
        nc = set()
        for t in range(1, 1+self.block_steps):
            nc |= set((p[0], t) for p in self.other_paths.values())
        solution, cost = spacetime_astar_ccr(self.g, source, goal, limit=self.limit, belief=self.belief, predecessors=pred, node_contraints=nc, preferred_nodes=preferred_nodes, inertia=self.inertia)
        assert len(solution) > 0
        assert solution[0] == source
        assert solution[-1] == goal
        return solution, cost

    def get_plan(self) -> list[int]:
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
        if self.quality_metric is None or self.quality_metric == "flow":
            fg = self.compute_flow_graph()
            qualities = self.compute_qualities_flow(fg, node, options)
            bs = BeliefState(state=node, priorities={e: q for e, q in qualities})
            #bs.normalize()
            # logging.warn(f"bs: {bs}")
            return bs
        elif self.quality_metric == "criticality":
            c_0 = discounted_criticalality(self.g, node, self.state, self.goal)
            bs = BeliefState(state=node, priorities={e: discounted_criticalality(self.g, e[0], self.state, self.goal) - c_0 + np.random.rand()*0.01 for e in options})
        elif self.quality_metric == "weighted_criticality":
            c_0 = discounted_criticalality(self.g, node, self.state, self.goal)
            bs_criticality = BeliefState(state=node, priorities={e: discounted_criticalality(self.g, e[0], self.state, self.goal) - c_0 for e in options})
            fg = self.compute_flow_graph()
            qualities = self.compute_qualities_flow(fg, node, options)
            bs_flow = BeliefState(state=node, priorities={e: q for e, q in qualities})
            return bs_criticality * 0.9 + bs_flow * 0.1
                
        else:
            raise NotImplementedError(f"{self.quality_metric} is not implemented")
            

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

from networkx.algorithms.connectivity import build_auxiliary_edge_connectivity
from networkx.algorithms.flow import build_residual_network

@lru_cache
def cached_HR(graph: nx.Graph):
    H = build_auxiliary_edge_connectivity(graph)
    return H, build_residual_network(graph, "capacity")

def discounted_criticalality(graph: nx.Graph, state:int, goal:int, discount_factor:float=0.5):
    path = nx.shortest_path(graph, source=state, target=goal, weight="weight")
    sum = 0
    for t, n in enumerate(path):
        sum += discount_factor ** t * criticality(graph, n, state, goal, **kwargs)
    return sum

@lru_cache
def criticality(graph: nx.Graph, state:int, start: int, goal:int, **kwargs):
    """Calculate discounted criticality of an edge in graph

    :param graph: roadmap graph
    :param state: starting node
    :param start: node 1
    :param goal: goal of the agent
    :param dicscount: discount factor, defaults to 0.5
    :return: returns the discounted criticality of the edge
    """
    
    # critcality is defined to what happens to the existance of a path, given the removal of the neighbourhing edges of the node
    # first find the path from start to goal, if the state is not on the path, criticality is 0
    path = nx.shortest_path(graph, source=start, target=goal, weight="weight")
    if state not in path:
        return 0.0
    # find all neighbours of the state
    H = build_auxiliary_edge_connectivity(graph)
    R = build_residual_network(H, "capacity")
    
    sum = 0
    for n in graph.neighbors(state):
        if n in path:
            ec = nx.local_edge_connectivity(graph, state, n, auxiliary=H, residual=R)
            if ec == 0:
                sum += 0.0
            else:
                sum += 1.0 / ec
    return sum
                        

class PriorityAgent:
    def __init__(self, graph: nx.Graph, state:int, goal:int, planning_problem_parameters, index: int, limit=100, inertia:float=0.2, block_steps=3, priorities=None):
        self.g = graph.copy()
        for node in self.g.nodes():
            self.g.add_edge(node, node, weight=planning_problem_parameters.wait_action_cost)
        self._state = state
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
        if state == self._state:
            return
        self.state_changed = True
        self._state = state
        #self.compute_plan()
        
    def update_goal(self, goal):
        if goal == self.goal:
            return
        self.goal_changed = True
        self.goal = goal
        self.compute_plan()

    def get_path(self, source, goal) -> tuple[list, float]:
        if source is None:
            return [None], np.inf
        preferred_nodes = set(self.plan)
        nc = set()
        for agent, path in self.other_paths.items():
            if agent not in self.priorities:
                self.priorities[agent] = 0
            # do not wait for other agents with lower priority
            if self.priorities[agent] < self.priorities[self.index]:
                continue
            horizon = self.planning_problem_parameters.conflict_horizon
            if horizon is None:
                horizon = len(path)
            for i, node in enumerate(path[:horizon]):
                # we need to check, that there is no other edge with more priority used for this node
                # if i==0, this causes a negative time in the constraint --- however, we do not plan for this so the constraint is not used
                nc.add((node, i-1))
                nc.add((node, i))
                nc.add((node, i+1))
                    
        # we are not allowed to go to the position of another robot at and t=1, because this will be a conflict that is not possible to be resolved from the other side
        for t in range(1, 1+self.block_steps):
            nc |= set((p[0], t) for p in self.other_paths.values())
        return spacetime_astar_ccr(self.g, source, goal, limit=self.limit, belief=None, predecessors=set(), node_contraints=nc, preferred_nodes=preferred_nodes, inertia=self.inertia)

    def get_plan(self) -> list[int]:
        #self.compute_plan()
        return self.plan
    
    def _get_conflicts(self, path):
        conflicts = compute_all_k_conflicts([self.plan, path], limit=self.planning_problem_parameters.conflict_horizon, k=self.planning_problem_parameters.k_robustness)
        self.conflicts = frozenset().union(conflicts, self.conflicts)
        return conflicts

    def get_conflicts(self):
        self.conflicts = frozenset()
        for _, path in self.other_paths.items():
            self._get_conflicts(path)
        return self.conflicts
    
    def replan_needed(self):
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
        if self._state is None:
            self.plan = [None]
            return
        self.plan, self.cost = self.get_path(self._state, self.goal)
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
            else:
                raise NotImplementedError
        
        self.agents = [
            PriorityAgent(self.g, sg[0], sg[1], self.environment.planning_problem_parameters, i, priorities=priorities)
            for i, sg in enumerate(self.environment.get_state_goal_tuples())
        ]
    
    def create_plan(self, *_) -> Plans:
        self.update_state(self.environment.state)
        self.planning_loop()
        # post-process plans to right format:
        plans = [agent.get_plan() for agent in self.agents]
        return Plans(plans)

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
        self.plans = {a.index: a.plan for a in self.agents}
        for a in self.agents:
            a.update_other_paths(self.plans)
    
    def make_all_plans_consistent(self):
        self.update_all_paths()
        # sort agents by priority to plan for highest prio first (saves tonnes of planning effort)
        for a in sorted(self.agents, key=lambda x: x.priorities[x.index], reverse=True):
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
        return random.choice(list(nodes))
    
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
                
    def create_plan(self, *_) -> Plans:
        """Create a plan for all agents
        
        :return: list of plans for all agents
        """
        self.update_state(self.environment.state)
        self.planning_loop()
        # post-process plans to right format:
        plans = [agent.get_plan() for agent in self.agents]
        return Plans(plans)

@lru_cache
def state_value_distance(graph: nx.Graph, goal: int, **kwargs):
    return {k: v for k, v in nx.single_target_shortest_path_length(graph, goal)}

@lru_cache
def criticality_matrix(graph: nx.Graph, goal: int, **kwargs):
    raise NotImplementedError

class StateValueAgent():
    graph: nx.Graph
    state: int
    goal: int
    planning_problem_parameters: PlanningProblemParameters
    other_state: dict = {}
    index: int
    

    def __init__(self, graph: nx.Graph, state:int, goal:int, planning_problem_parameters, value_weights:dict={}, index:int=0) -> None:
        self.graph = graph
        self.state = state
        self.goal = goal
        self.planning_problem_parameters = planning_problem_parameters
        self.value_weights = value_weights
        if not self.value_weights:
            self.value_weights = {"state_value_distance": -1.0}
        self.index = index
    
    def value_function(self):
        s = {n : 0 for n in self.graph.nodes()}
        
        for value_function, weight in self.value_weights.items():
            if value_function == "state_value_distance":
                values = state_value_distance(self.graph, self.goal)
            elif value_function == "criticality_matrix":
                values = criticality_matrix(self.graph, self.goal)
            elif value_function == "agent_plan_penalty":
                # value function that puts a penalty on other agents plans / values whatever
                raise NotImplementedError
            else:
                raise NotImplementedError
            for n, v in values.items():
                s[n] += v * weight
            pass
        return s

    def update_other_state(self, states):
        self.other_state |= states
        if self.index in self.other_state:
            del self.other_state[self.index]
        
    def update_state(self, state):
        self.state = state
        
    def get_state_values(self):
        return self.value_function()
    
    def get_plan(self) -> Plans:
        plan = [self.state]
        while self.state is not None and self.state != self.goal:
            plan.append( self.action_selection(state=plan[-1]) )
            step_num = self.planning_problem_parameters.step_num if self.planning_problem_parameters.step_num else 10
            if len(plan) >= step_num:
                break
        return plan
        

    def action_selection(self, action_set: list=None, state=None) -> Plans:
        if state is None:
            state = self.state
        # get action set, i.e., the current state and its neighbours in the graph
        if action_set is None:
            action_set = set(self.graph.neighbors(state))
            action_set |= {state}
            action_set = list(action_set)
        if len(action_set) == 1:
            return action_set[0]
        
        # remove states of other agents from action set
        action_set = [s for s in action_set if s is not None and s not in self.other_state.values()]

        # get state values
        state_values = self.get_state_values()        
        action_values = {a: state_values[a] for a in action_set}

        # return best action
        return max(action_values, key=action_values.get)

class StateValueAgentPlanner(Planner):
    environment: Environment
    agents: list[StateValueAgent]

    def __init__(self, environment: Environment, **kwargs) -> None:
        # create agents
        self.agents = []
        self.environment = environment
        for i, sg in enumerate(environment.get_state_goal_tuples()):
            self.agents.append(StateValueAgent(environment.g, sg[0], sg[1], environment.planning_problem_parameters, index=i, **kwargs))

        super.__init__(environment, **kwargs)

    def create_plan(self, *_) -> Plans:
        states = enumerate(self.environment.state)
        for agent in self.agents:
            agent.update_other_state(states)
        return Plans([agent.get_plan() for agent in self.agents])


# implement learning based planners
class LearningAgent:
    alpha:float
    beta:float
    gamma:float
    epsilon:float
    evaporation_rate:float
    dispersion_rate:float
    method:str
    episodes:int
    graph:nx.Graph

    def __init__(self,
                 graph:nx.Graph,
                 start,
                 goal,
                 nodelist,
                 alpha:float=1.0,
                 beta:float=1.0,
                 gamma:float=1,
                 delta:float=0.1,
                 epsilon:float=0.3,
                 evaporation_rate:float=0.1,
                 dispersion_rate:float=0.1,
                 collision_weight:float=0.8,
                 method:str="aco",
                 episodes:int=10,
                 time_horizon:int=10) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.evaporation_rate = evaporation_rate
        self.dispersion_rate = dispersion_rate
        self.collision_weight = collision_weight
        self.method = method
        self.time_horizon = time_horizon
        self.episodes = episodes
        self.start = start
        self.goal = goal
        self.G = graph
        self.nodelist = nodelist
        self.G_t = self.create_Gt()
        self._occupancy = np.zeros((len(self.nodelist), self.time_horizon))
        self.latest_episodes = []
        
        if method == "aco":
            self.q_function = self.aco_decsion_values
            self.selection_function = self.epsilon_fitness_proportional
            self.update_q_function = self.update_q_aco
            self._other_pheromones = np.zeros((len(self.nodelist), self.time_horizon))

        elif method == "q":
            pass
        elif method == "sq":
            self.q_function = self.sq_decsion_values
            self.selection_function = self.epsilon_greedy
            self.update_q_function = self.update_q_sq
        else:
            raise NotImplementedError
        shortest = nx.shortest_path(self.G, self.start, self.goal)
        assert shortest is not None


    def set_occupancy(self, occupancy):
        self._occupancy = occupancy
        self._occupancy[:,:-1] += occupancy[:,1:]
        self._occupancy[:,1:] += occupancy[:,:-1]
        
    def set_other_pheromones(self, other_pheromones):
        self._other_pheromones = other_pheromones
        # smear the other pheromones to adjacent time-steps, such that we avoid those time-steps also
        self._other_pheromones[:,:-1] += other_pheromones[:,1:]
        self._other_pheromones[:,1:] += other_pheromones[:,:-1]
        
    def get_occupancy(self):
        occupancy = np.zeros_like(self._occupancy)
        for e in self.latest_episodes:
            for t, n in enumerate(e[:self.time_horizon]):
                occupancy[self.nodelist.index(n), t] += 1
        return occupancy

    def update_state(self, state):
        self.start = state
        # reset Q-values
        self._occupancy = np.zeros((len(self.nodelist), self.time_horizon))
        nx.set_edge_attributes(self.G_t, 1.0, 'Q')

    def update_goal(self, goal):
        self.goal = goal
        self.heuristic.cache_clear()
        # reset Q-values
        self._occupancy = np.zeros((len(self.nodelist), self.time_horizon))
        nx.set_edge_attributes(self.G_t, 1.0, 'Q')

    def create_Gt(self):
        G_t = nx.DiGraph()
        for t in range(self.time_horizon):
            for v in self.nodelist:
                G_t.add_node((v, t))
                if t > 0:
                    G_t.add_edge((v, t-1), (v, t))  # Wait action
                    for neighbor in self.G.neighbors(v):
                        G_t.add_edge((v, t-1), (neighbor, t))  # Move action
        nx.set_edge_attributes(G_t, 1.0, 'Q')
        return G_t
    
    def run_episodes(self, i=0):
        episodes = []
        for _ in range(self.episodes):
            state, t = self.start, 0
            episode = [state]
            while state is not None:
                values = self.q_function(state, t)
                state = self.selection_function(values)
                episode.append(state)
                t += 1
                if t >= self.time_horizon:
                    break
                if state == self.goal:
                    break
            # fill episodes with shortes path, if time horizon is reached
            if state is None:
                episode = episode[:-1]
                if not len(episode):
                    continue
            if episode[-1] != self.goal:
                episode = episode + list(nx.shortest_path(self.G, episode[-1], self.goal))
            episodes.append(episode)
        self.latest_episodes = episodes
        return episodes
    
    def iteration(self, i=0):
        episodes = self.run_episodes(i=i)
        self.update_q_function(episodes)
    
    def aco_other_pheromones(self, state, t):
        return self._other_pheromones[self.nodelist.index(state), t]
    
    @lru_cache(maxsize=None)
    def heuristic(self, state):
        return nx.shortest_path_length(self.G, state, self.goal)

    def aco_decsion_values(self, state, t):
        neighbours = list(self.G_t.neighbors((state, t)))
        if len(neighbours) == 0:
            return {}
        values = {}
        for n in neighbours:
            alpha = self.G_t[(state, t)][n]['Q'] ** self.alpha
            beta = 1 / (1 + self.heuristic(n[0])) ** self.beta
            gamma = 1 / (1 + self.aco_other_pheromones(*n)) ** self.gamma
            delta = 1 / (1 + self._occupancy[self.nodelist.index(n[0]), n[1]]) ** self.delta
            values[n[0]] = alpha * beta * gamma * delta
            if np.isnan(values[n[0]]):
                values[n[0]] = 0
                logging.warning(f"NAN-value in decsion function: {alpha:.2f},{beta:.2f},{gamma:.2f},{delta:.2f} = {values[n[0]]:.2f}")
            #if delta!=1.0:
            #    print("delta != 1.0")
            #    print(f"n: {n[0]}, t: {n[1]}")
            #    print(f"{alpha:.2f},{beta:.2f},{gamma:.2f},{delta:.2f} = {values[n[0]]:.2f}")
        return values

    def sq_decsion_values(self, state, t):
        neighbours = list(self.G_t.neighbors((state, t)))
        if len(neighbours) == 0:
            return {}
        values = {}
        for n in neighbours:
            Q = self.G_t[(state, t)][n]['Q']
            
            values[n[0]] = Q 
        return values
    
    def fitness_proportional(self, values):
        if not len (values):
            return None
        s = sum(values.values())
        values = {k: v / s for k, v in values.items()}
        decision = np.random.choice(list(values.keys()), p=list(values.values()))
        return decision
    
    def greedy(self, values):
        #print("greedy")
        if not len (values):
            return None
        return max(values, key=values.get)
    
    def epsilon_greedy(self, values):
        if not len (values):
            return None
        if np.random.rand() < self.epsilon:
            return self.random_selection(values)
        return self.greedy(values)
    
    def epsilon_fitness_proportional(self, values):
        if not len(values):
            return None
        if np.random.rand() < self.epsilon:
            return self.random_selection(values)
        return self.fitness_proportional(values)
    
    def random_selection(self, values):
        if not len (values):
            return None
        return np.random.choice(list(values.keys()))
    
    def get_plan(self) -> list:
        state = self.start
        t = 0
        plan = [state]
        for _ in range(self.time_horizon-1):
            values = self.q_function(state, t)
            state = self.greedy(values)
            t += 1
            plan.append(state)
            if state == self.goal:
                break
        return plan[:self.time_horizon]
    
    def get_episodes(self) -> list:
        return self.latest_episodes
    
    def objetcive_length(self, path:list):
        return len(path)
    
    def normalize_objective(self, values):
        max_values = max(values)
        min_values = min(values)
        if min_values == max_values:
            return [1.0 for _ in values]
        return [(e- min_values) / (max_values - min_values) for e in values]
    
    def objetcive_collision_prob(self, path:list):
        if not path:
            return 0
        collisions = 0
        for t, n in enumerate(path[:self.time_horizon]):
            collisions += self._occupancy[self.nodelist.index(n), t]
        return collisions / len(path)

    def update_q_aco(self, episodes:list):
        # evaporation
        for u, v in self.G_t.edges():
            self.G_t[u][v]['Q'] *= (1 - self.evaporation_rate)

        # pheromone deposition
        # I should test if this actually improves things
        # Only use unique episodes:
        episodes = list(set([tuple(e) for e in episodes]))
        # 0: shortest path, 1: longest path
        # 0 is better
        normalized_length = self.normalize_objective([self.objetcive_length(e) for e in episodes])
        # 0: no collisions, 1: max collision prob
        # 0 is better
        normalized_collision_prob = self.normalize_objective([self.objetcive_collision_prob(e) for e in episodes])
        # for combined score we have to make sure that we have a maximization obejective, i.e., 1-length, 1-collision_prob
        combined_score = normalized_length # (1-self.collision_weight) * (1 - np.array(normalized_length)) + self.collision_weight * (1 - np.array(normalized_collision_prob))
        
        #print(combined_score)
        for episode, score in zip(episodes, combined_score):
            # add pheromones to the edge s_t -> s_{t+1}
            for s_p, s_n, t in zip(episode[:-1], episode[1:], range(0, len(episode)-1)):
                if t+1 >= self.time_horizon:
                    break
                self.G_t[(s_p, t)][(s_n, t+1)]['Q'] += score
        
        # dispersion -- not implemented
        self.disperse_pheromone()
        
    def get_q_matrix(self):
        # we dont save the q-values themselves, but the ingoing values for each state
        Q = np.zeros((len(self.nodelist), self.time_horizon))
        for u, v in self.G_t.edges():
            Q[self.nodelist.index(v[0]), v[1]] = self.G_t[u][v]['Q']
        return Q

    def disperse_pheromone(self):
        # forward and backward dispersion of pheromones
        node_pheromones = np.zeros((len(self.nodelist), self.time_horizon))
        # gather pheromone values in adjacent nodes
        for u, v in self.G_t.edges():
            node_pheromones[self.nodelist.index(v[0]), v[1]] += self.G_t[u][v]['Q']
            node_pheromones[self.nodelist.index(u[0]), u[1]] += self.G_t[u][v]['Q']
        # add pheromones to the edges adjacent to each node
        for u, v in self.G_t.edges():
            self.G_t[u][v]['Q'] += self.dispersion_rate * node_pheromones[self.nodelist.index(u[0]), u[1]]
            self.G_t[u][v]['Q'] += self.dispersion_rate * node_pheromones[self.nodelist.index(v[0]), v[1]]
            
    def update_simplified_q(self, episodes:list):
        pass
    
    def update_q(self, episodes:list):
        pass


class LearningAgentPlanner(Planner):
    def __init__(self,
                 environment: Environment,
                 alpha:float=1,
                 beta:float=1,
                 gamma:float=0.5,
                 delta:float=0.5,
                 epsilon:float=0.5,
                 evaporation_rate:float=0.1, 
                 dispersion_rate:float=0.001,
                 collision_weight:float=0.5,
                 method:str="aco",
                 episodes:int=25,
                 iterations=50,
                 **kwargs) -> None:
        # we need a nodelist, so we have a determistic ordering of the nodes of G
        #super.__init__(self, environment, **kwargs)

        self.method = method
        self.environment = environment
        self.iterations = iterations
        self.nodelist = list(self.environment.g.nodes())
        self.horizon = self.environment.planning_horizon
        self.epsilon = epsilon
        if environment.planning_problem_parameters.conflict_horizon is not None:
            self.horizon = environment.planning_problem_parameters.conflict_horizon
        self.agents = [
            LearningAgent(
                self.environment.g,
                s, g,
                self.nodelist,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                delta=delta,
                epsilon=epsilon,
                evaporation_rate=evaporation_rate,
                dispersion_rate=dispersion_rate,
                collision_weight=collision_weight,
                method=method,
                episodes=episodes,
                time_horizon=self.horizon)
            for s, g in self.environment.get_state_goal_tuples()]
        
    def run(self):
        self.communicate()
        for i in range(self.iterations):
            for agent in self.agents:
                agent.epsilon = self.epsilon * (i / self.iterations)
                agent.iteration()
                self.communicate()
        return [agent.get_plan() for agent in self.agents]
    
    def communicate(self):
        occupancy = sum([agent.get_occupancy() for agent in self.agents])
        for agent in self.agents:
            agent.set_occupancy((occupancy - agent.get_occupancy()) / (len(self.agents)-1))
            
        if self.method != "aco":
            return

        all_node_pheromones = sum([agent.get_q_matrix() for agent in self.agents])
        for agent in self.agents:
            agent.set_other_pheromones((all_node_pheromones - agent.get_q_matrix()) / (len(self.agents)-1))


    def create_plan(self, *_) -> Plans:
        self.run()
        plans = Plans([agent.get_plan() for agent in self.agents])
        if not plans.is_valid(self.environment):
            print(plans)
        #    raise nx.NetworkXNoPath
        return plans
    
    def update_state(self, state):
        self.environment.state = state
        for s, agent in zip(state, self.agents):
            agent.update_state(s)


if __name__ == "__main__":
    pass

