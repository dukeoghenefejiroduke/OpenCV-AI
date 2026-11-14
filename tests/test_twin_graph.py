"""Unit tests for Phase 3 Digital Twin Graph & Health Propagation (tests/test_twin_graph.py)."""

import pytest
from twin.graph import DigitalTwinGraph


def test_valid_graph_and_health_propagation():
    graph_data = {
        "namespace": "SIM",
        "nodes": [
            {"id": "node_A", "health_status": "CRITICAL"},
            {"id": "node_B", "health_status": "HEALTHY"}
        ],
        "segments": [
            {"source": "node_A", "target": "node_B"}
        ],
        "assets": [
            {"id": "SIM-PIPE-01", "health_status": "CRITICAL"}
        ]
    }
    twin = DigitalTwinGraph(graph_data)
    assert twin.is_simulated is True

    result = twin.aggregate_and_propagate_health()
    assert result["namespace"] == "SIM"
    # node_A (CRITICAL) propagates to node_B (HEALTHY), so node_B becomes CRITICAL
    assert result["propagated_health"]["node_A"] == "CRITICAL"
    assert result["propagated_health"]["node_B"] == "CRITICAL"


def test_malformed_graph_rejection():
    # Missing nodes key
    with pytest.raises(ValueError, match="Malformed graph"):
        DigitalTwinGraph({"segments": [], "assets": []})

    # Segment referencing non-existent node
    with pytest.raises(ValueError, match="Malformed segment"):
        DigitalTwinGraph({
            "nodes": [{"id": "node_A"}],
            "segments": [{"source": "node_A", "target": "node_X"}],
            "assets": []
        })


def test_cycle_detection():
    cyclic_graph = {
        "namespace": "SIM",
        "nodes": [
            {"id": "n1"},
            {"id": "n2"}
        ],
        "segments": [
            {"source": "n1", "target": "n2"},
            {"source": "n2", "target": "n1"}
        ],
        "assets": []
    }
    with pytest.raises(ValueError, match="cyclic dependency"):
        DigitalTwinGraph(cyclic_graph)


def test_namespace_isolation():
    graph_data = {
        "namespace": "SIM",
        "nodes": [{"id": "SIM-n1"}],
        "segments": [],
        "assets": [{"id": "SIM-asset-01"}]
    }
    twin = DigitalTwinGraph(graph_data)
    real_workspace = {"SIM-asset-01", "REAL-ASSET-02"}
    with pytest.raises(ValueError, match="Namespace isolation violation"):
        twin.enforce_namespace_isolation(real_workspace)
