"""CementSight inspection workspace layer.

Sits ABOVE vision/orchestrator.py. Consumes MultiHazardOrchestrator results
read-only. Never modifies vision/inspection.py, never alters the authoritative
corrosion decision, never dispatches maintenance.
"""

__version__ = "1.0.0"