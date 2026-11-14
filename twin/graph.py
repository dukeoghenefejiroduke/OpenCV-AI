"""Digital Twin Graph module for CementSight AI (twin/graph.py)."""

from typing import Dict, Any, List, Set


class DigitalTwinGraph:
    """Manages digital twin nodes, connectivity segments, health propagation, and namespace isolation."""

    def __init__(self, graph_data: Dict[str, Any]):
        if not isinstance(graph_data, dict) or "nodes" not in graph_data:
            raise ValueError("Malformed graph: missing 'nodes' key.")
        
        self.namespace = graph_data.get("namespace", "SIM")
        self.nodes = graph_data.get("nodes", [])
        self.segments = graph_data.get("segments", [])
        self.assets = graph_data.get("assets", [])
        self.is_simulated = True

        node_ids = {n["id"] for n in self.nodes if isinstance(n, dict) and "id" in n}
        
        # Validate segments
        for seg in self.segments:
            src = seg.get("source")
            tgt = seg.get("target")
            if src not in node_ids or tgt not in node_ids:
                raise ValueError(f"Malformed segment referencing non-existent node: {src} -> {tgt}")

        # Check for cycles (simple DFS)
        self._check_cycles(node_ids)

    def _check_cycles(self, node_ids: Set[str]):
        adj = {nid: [] for nid in node_ids}
        for seg in self.segments:
            adj[seg.get("source")].append(seg.get("target"))

        visited = set()
        rec_stack = set()

        def dfs(node):
            visited.add(node)
            rec_stack.add(node)
            for neighbor in adj.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            rec_stack.remove(node)
            return False

        for n in node_ids:
            if n not in visited:
                if dfs(n):
                    raise ValueError("Malformed graph: cyclic dependency detected.")

    def aggregate_and_propagate_health(self) -> Dict[str, Any]:
        """Propagates health status (CRITICAL overrides WARNING/HEALTHY) along segments."""
        health_map = {n["id"]: n.get("health_status", "HEALTHY") for n in self.nodes}

        # Severity ranking
        severity = {"CRITICAL": 3, "WARNING": 2, "HEALTHY": 1}

        # Build adjacency for propagation
        adj = {nid: [] for nid in health_map}
        for seg in self.segments:
            adj[seg.get("source")].append(seg.get("target"))

        # Propagate from sources to targets
        propagated = dict(health_map)
        changed = True
        while changed:
            changed = False
            for src, targets in adj.items():
                src_status = propagated.get(src, "HEALTHY")
                for tgt in targets:
                    tgt_status = propagated.get(tgt, "HEALTHY")
                    if severity.get(src_status, 1) > severity.get(tgt_status, 1):
                        propagated[tgt] = src_status
                        changed = True

        return {
            "namespace": self.namespace,
            "badge": "[SIMULATED]",
            "node_count": len(self.nodes),
            "segment_count": len(self.segments),
            "initial_health": health_map,
            "propagated_health": propagated
        }

    def enforce_namespace_isolation(self, real_workspace_assets: Set[str]):
        """Ensures simulation namespace does not contaminate real workspace assets."""
        for asset in self.assets:
            aid = asset.get("id") if isinstance(asset, dict) else asset
            if aid in real_workspace_assets:
                raise ValueError(f"Namespace isolation violation: simulated asset {aid} overlaps with real workspace.")
        for node in self.nodes:
            nid = node.get("id")
            if nid in real_workspace_assets:
                raise ValueError(f"Namespace isolation violation: simulated node {nid} overlaps with real workspace.")
