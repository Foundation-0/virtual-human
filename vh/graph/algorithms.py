"""Graph algorithms that answer structural questions about the formalized body.

All operate on a `Graph` (or an `edge_view` of one, e.g. the blood-flow subgraph
or the bond subgraph). Implemented from scratch: reachability, path enumeration,
cycle / strongly-connected components, topological sort, connected components,
widest (dominant-flux) path, flow mass-balance, and degree/betweenness centrality.

The point: once the body is a typed graph, these are the exact answers to
questions like "where can this drug reach", "what path does it take", "is this
system mass-balanced", "where is the bottleneck", "is there recirculation".
"""

from __future__ import annotations

import heapq
from collections import defaultdict, deque
from typing import Iterable

from .core import Graph


# ---------------------------------------------------------------------------
# reachability and paths
# ---------------------------------------------------------------------------
def reachable(g: Graph, source: str, edge_types: Iterable[str] | None = None) -> set[str]:
    """Every node reachable from `source` following edges of the given types."""
    seen, stack = {source}, [source]
    while stack:
        u = stack.pop()
        for v in g.successors(u, edge_types):
            if v not in seen:
                seen.add(v)
                stack.append(v)
    seen.discard(source)
    return seen


def bfs_distances(g: Graph, source: str, edge_types: Iterable[str] | None = None) -> dict[str, int]:
    """Hop count from source to every reachable node (undirected or directed as typed)."""
    dist = {source: 0}
    q = deque([source])
    while q:
        u = q.popleft()
        for v in g.successors(u, edge_types):
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def all_paths(g: Graph, source: str, target: str, edge_types: Iterable[str] | None = None,
              max_length: int = 12) -> list[list[str]]:
    """All simple paths source -> target (node id lists), up to max_length edges."""
    paths: list[list[str]] = []

    def dfs(u, path, visited):
        if len(path) - 1 > max_length:
            return
        if u == target and len(path) > 1:
            paths.append(list(path))
            return
        for v in g.successors(u, edge_types):
            if v not in visited:
                visited.add(v)
                path.append(v)
                dfs(v, path, visited)
                path.pop()
                visited.discard(v)

    dfs(source, [source], {source})
    return paths


# ---------------------------------------------------------------------------
# cycles / strongly connected components (recirculation, feedback)
# ---------------------------------------------------------------------------
def strongly_connected_components(g: Graph, edge_types: Iterable[str] | None = None) -> list[list[str]]:
    """Tarjan's SCC. A component with >1 node (or a self-loop) means the drug/signal
    can recirculate through those nodes."""
    index = {}
    low = {}
    on_stack = {}
    stack: list[str] = []
    out: list[list[str]] = []
    counter = [0]

    def strong(v):
        index[v] = low[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        on_stack[v] = True
        for w in g.successors(v, edge_types):
            if w not in index:
                strong(w)
                low[v] = min(low[v], low[w])
            elif on_stack.get(w):
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                comp.append(w)
                if w == v:
                    break
            out.append(comp)

    for v in g.nodes:
        if v not in index:
            strong(v)
    return out


def recirculation_loops(g: Graph, edge_types: Iterable[str] | None = None) -> list[list[str]]:
    """Nontrivial strongly-connected components: the parts where flow recirculates."""
    sccs = strongly_connected_components(g, edge_types)
    return [c for c in sccs if len(c) > 1]


def simple_cycles(g: Graph, edge_types: Iterable[str] | None = None, max_cycles: int = 1000) -> list[list[str]]:
    """Enumerate simple directed cycles (DFS; for the modestly sized graphs here)."""
    cycles: list[list[str]] = []
    nodes = list(g.nodes)
    order = {n: i for i, n in enumerate(nodes)}

    def dfs(start, u, path, visited):
        if len(cycles) >= max_cycles:
            return
        for v in g.successors(u, edge_types):
            if v == start and len(path) >= 1:
                cycles.append(path + [start])
            elif order.get(v, -1) > order[start] and v not in visited:
                visited.add(v)
                dfs(start, v, path + [v], visited)
                visited.discard(v)

    for s in nodes:
        dfs(s, s, [s], {s})
    return cycles


# ---------------------------------------------------------------------------
# ordering and components
# ---------------------------------------------------------------------------
def topological_sort(g: Graph, edge_types: Iterable[str] | None = None) -> list[str] | None:
    """Kahn's algorithm; returns None if the (typed) graph has a cycle."""
    indeg = {n: 0 for n in g.nodes}
    for u in g.nodes:
        for v in g.successors(u, edge_types):
            indeg[v] += 1
    q = deque([n for n, d in indeg.items() if d == 0])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in g.successors(u, edge_types):
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return order if len(order) == len(g.nodes) else None


def connected_components(g: Graph, edge_types: Iterable[str] | None = None) -> list[list[str]]:
    """Weakly connected components (edges treated as undirected)."""
    parent = {n: n for n in g.nodes}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    ts = set(edge_types) if edge_types is not None else None
    for e in g.edges:
        if ts is None or e.type in ts:
            ra, rb = find(e.src), find(e.dst)
            if ra != rb:
                parent[ra] = rb
    comps = defaultdict(list)
    for n in g.nodes:
        comps[find(n)].append(n)
    return list(comps.values())


# ---------------------------------------------------------------------------
# weighted paths (dominant flux route)
# ---------------------------------------------------------------------------
def widest_path(g: Graph, source: str, target: str, weight: str, edge_types: Iterable[str] | None = None):
    """Max-min (bottleneck) path: the route whose smallest edge `weight` is largest.
    For a flow network this is the dominant route a tracer takes."""
    best = {source: float("inf")}
    prev = {}
    pq = [(-float("inf"), source)]
    while pq:
        negw, u = heapq.heappop(pq)
        w = -negw
        if u == target:
            break
        if w < best.get(u, float("-inf")):
            continue
        for e in g.out_edges(u, edge_types):
            cap = min(w, float(e.attrs.get(weight, 0.0)))
            if cap > best.get(e.dst, float("-inf")):
                best[e.dst] = cap
                prev[e.dst] = u
                heapq.heappush(pq, (-cap, e.dst))
    if target not in prev and target != source:
        return None, 0.0
    path = [target]
    while path[-1] != source:
        path.append(prev[path[-1]])
    return path[::-1], best.get(target, 0.0)


# ---------------------------------------------------------------------------
# flow conservation (a verifiable physical law on the graph)
# ---------------------------------------------------------------------------
def flow_balance(g: Graph, flow_attr: str = "flow", edge_types=("flow",),
                 sources: Iterable[str] = (), sinks: Iterable[str] = ()) -> dict:
    """Kirchhoff's law on the flow subgraph: at steady state, total flow into each
    compartment equals total flow out (except designated sources/sinks).

    Returns per-node residual (in - out) and the worst imbalance -- a structural
    check that the formalization obeys conservation of blood (or mass).
    """
    src, snk = set(sources), set(sinks)
    inflow = defaultdict(float)
    outflow = defaultdict(float)
    for e in g.edges_of_type(*edge_types):
        f = float(e.attrs.get(flow_attr, 0.0))
        outflow[e.src] += f
        inflow[e.dst] += f
    residual = {}
    for n in g.nodes:
        if n in src or n in snk:
            continue
        residual[n] = inflow[n] - outflow[n]
    worst = max((abs(r) for r in residual.values()), default=0.0)
    scale = max((abs(v) for v in list(inflow.values()) + list(outflow.values())), default=1.0)
    return {"residual": residual, "max_abs_residual": worst,
            "max_rel_residual": worst / scale if scale else 0.0,
            "balanced": worst / scale < 1e-9 if scale else True}


# ---------------------------------------------------------------------------
# centrality (hubs and bottlenecks)
# ---------------------------------------------------------------------------
def degree_centrality(g: Graph, edge_types: Iterable[str] | None = None) -> dict[str, int]:
    deg = {n: 0 for n in g.nodes}
    ts = set(edge_types) if edge_types is not None else None
    for e in g.edges:
        if ts is None or e.type in ts:
            deg[e.src] += 1
            deg[e.dst] += 1
    return deg


def betweenness_centrality(g: Graph, edge_types: Iterable[str] | None = None) -> dict[str, float]:
    """Brandes' algorithm (unweighted). High betweenness = a node many shortest
    paths pass through: a structural bottleneck / hub."""
    bc = {n: 0.0 for n in g.nodes}
    for s in g.nodes:
        stack = []
        pred = {w: [] for w in g.nodes}
        sigma = dict.fromkeys(g.nodes, 0.0)
        sigma[s] = 1.0
        dist = {s: 0}
        q = deque([s])
        while q:
            v = q.popleft()
            stack.append(v)
            for w in g.successors(v, edge_types):
                if w not in dist:
                    dist[w] = dist[v] + 1
                    q.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)
        delta = dict.fromkeys(g.nodes, 0.0)
        while stack:
            w = stack.pop()
            for v in pred[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
            if w != s:
                bc[w] += delta[w]
    return bc


# ---------------------------------------------------------------------------
# weighted flow networks: Laplacian, equivalent resistance, current flow
# ---------------------------------------------------------------------------
def weighted_laplacian(g: Graph, weight: str = "conductance",
                       edge_types: Iterable[str] | None = None):
    """Graph Laplacian for a resistive/conductive network.

    Each edge contributes its ``weight`` (default: conductance, 1/R) to the
    off-diagonal and to both diagonal entries.  Directed edges are treated as
    symmetric conductors for the equivalent-resistance calculation.  Returns
    ``(L, node_ids)``.
    """
    import numpy as np

    nodes = list(g.nodes)
    ix = {n: i for i, n in enumerate(nodes)}
    L = np.zeros((len(nodes), len(nodes)))
    types = set(edge_types) if edge_types is not None else None
    for e in g.edges:
        if types is not None and e.type not in types:
            continue
        c = float(e.attrs.get(weight, 0.0))
        if c <= 0:
            continue
        i, j = ix[e.src], ix[e.dst]
        if i == j:
            continue
        L[i, i] += c
        L[j, j] += c
        L[i, j] -= c
        L[j, i] -= c
    return L, nodes


def effective_resistance(g: Graph, source: str, sink: str,
                         weight: str = "conductance",
                         edge_types: Iterable[str] | None = None) -> float:
    """Equivalent resistance between two nodes, in the inverse units of ``weight``.

    This is the graph-Laplacian answer to "what single resistor would replace
    this whole vein network?":

        R_eff = (e_s - e_t)^T L^+ (e_s - e_t).

    For series edges R_eff = R1 + R2; for parallel edges
    R_eff = R1 R2 / (R1 + R2).  It is computed from topology and the edge
    conductances alone, without solving the flow field.
    """
    import numpy as np

    if source not in g.nodes or sink not in g.nodes:
        raise KeyError("source and sink must be graph nodes")
    if source == sink:
        return 0.0
    comps = connected_components(g, edge_types)
    if not any(source in c and sink in c for c in comps):
        return float("inf")
    L, nodes = weighted_laplacian(g, weight, edge_types)
    ix = {n: i for i, n in enumerate(nodes)}
    Lp = np.linalg.pinv(L)
    i, j = ix[source], ix[sink]
    r = Lp[i, i] + Lp[j, j] - 2.0 * Lp[i, j]
    return float(max(r, 0.0))


def current_flow(g: Graph, source: str, sink: str,
                 weight: str = "conductance",
                 edge_types: Iterable[str] | None = None) -> dict:
    """Edge currents for unit injection at ``source`` and extraction at ``sink``.

    Solves L p = e_source - e_sink, then i_e = conductance_e (p_src - p_dst).
    The currents sum to 1 at the source.  Edges with the largest current are
    the main flow paths -- a physically weighted bottleneck query, sharper
    than unweighted betweenness for a flow network.
    """
    import numpy as np

    if source not in g.nodes or sink not in g.nodes:
        raise KeyError("source and sink must be graph nodes")
    L, nodes = weighted_laplacian(g, weight, edge_types)
    ix = {n: i for i, n in enumerate(nodes)}
    b = np.zeros(len(nodes))
    b[ix[source]] = 1.0
    b[ix[sink]] = -1.0
    p = np.linalg.pinv(L) @ b
    out: dict = defaultdict(float)
    types = set(edge_types) if edge_types is not None else None
    for e in g.edges:
        if types is not None and e.type not in types:
            continue
        c = float(e.attrs.get(weight, 0.0))
        if c <= 0:
            continue
        out[e.key] += c * (p[ix[e.src]] - p[ix[e.dst]])
    return dict(out)
