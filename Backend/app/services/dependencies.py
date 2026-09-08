from collections.abc import Awaitable, Callable

Probe = Callable[[], Awaitable[bool]]


class DependencyHealthService:
    """Reports external dependency readiness without hiding degraded services."""

    def __init__(self, postgres_probe: Probe, qdrant_probe: Probe, neo4j_probe: Probe) -> None:
        self._probes = {
            "postgres": postgres_probe,
            "qdrant": qdrant_probe,
            "neo4j": neo4j_probe,
        }

    async def check_all(self) -> dict[str, str]:
        results: dict[str, str] = {}
        for name, probe in self._probes.items():
            try:
                results[name] = "ready" if await probe() else "unavailable"
            except Exception:
                results[name] = "unavailable"
        return results
