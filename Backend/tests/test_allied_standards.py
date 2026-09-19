import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.services.allied_standards import (
    AlliedStandardNotFound,
    AlliedStandardRelation,
    AlliedStandardsGraphService,
)


class FakeNeo4jResult:
    def __init__(self, rows):
        self._rows = rows

    async def data(self):
        return self._rows


class FakeNeo4jSession:
    def __init__(self, driver):
        self._driver = driver

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None

    async def run(self, query, **params):
        if self._driver.unavailable:
            raise RuntimeError("neo4j unavailable")
        if "relations" in params:
            return FakeNeo4jResult([self._driver.load(params["relations"])])
        return FakeNeo4jResult(self._driver.lookup(params["standard_id"]))


class FakeNeo4jDriver:
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.unavailable = False

    def session(self):
        return FakeNeo4jSession(self)

    def add_node(self, standard_id, title=None):
        self.nodes[standard_id] = {"standard_id": standard_id, "title": title}

    def load(self, relations):
        for relation in relations:
            source = relation["source_standard_id"]
            target = relation["target_standard_id"]
            self.add_node(source)
            self.add_node(target, relation.get("target_title"))
            self.edges[(source, relation["relation_type"], target)] = {
                "source_standard_id": source,
                "target_standard_id": target,
                "target_title": relation.get("target_title"),
                "relation_type": relation["relation_type"],
                "evidence": relation.get("evidence"),
            }
        edge_count = len(self.edges)
        return {"nodes_merged": len(self.nodes), "edges_merged": edge_count}

    def lookup(self, standard_id):
        if standard_id not in self.nodes:
            return []
        rows = []
        for (source, relation_type, target), edge in sorted(self.edges.items()):
            if source == standard_id:
                rows.append(
                    {
                        "source_standard_id": source,
                        "target_standard_id": target,
                        "target_title": self.nodes[target].get("title"),
                        "relation_type": relation_type,
                        "evidence": edge.get("evidence"),
                    }
                )
            elif target == standard_id:
                rows.append(
                    {
                        "source_standard_id": target,
                        "target_standard_id": source,
                        "target_title": self.nodes[source].get("title"),
                        "relation_type": relation_type,
                        "evidence": edge.get("evidence"),
                    }
                )
        return rows or [
            {
                "source_standard_id": standard_id,
                "target_standard_id": None,
                "target_title": None,
                "relation_type": None,
                "evidence": None,
            }
        ]


class FakeNeo4jClientService:
    def __init__(self, driver):
        self.driver = driver

    async def close(self):
        return None


@pytest.mark.asyncio
async def test_known_standard_with_allied_edges_returns_relations():
    driver = FakeNeo4jDriver()
    service = AlliedStandardsGraphService(driver)
    await service.load_relations(
        [
            AlliedStandardRelation(
                source_standard_id="IS 123:2020",
                target_standard_id="IS 456:2000",
                target_title="Concrete",
                relation_type="references",
                evidence="scope",
            )
        ]
    )

    relations = await service.get_allied("is 123:2020")

    assert relations == [
        AlliedStandardRelation(
            source_standard_id="IS 123:2020",
            target_standard_id="IS 456:2000",
            target_title="Concrete",
            relation_type="REFERENCES",
            evidence="scope",
        )
    ]


@pytest.mark.asyncio
async def test_known_standard_with_no_graph_edges_returns_empty_list():
    driver = FakeNeo4jDriver()
    driver.add_node("IS 123:2020")
    service = AlliedStandardsGraphService(driver)

    assert await service.get_allied("IS 123:2020") == []


@pytest.mark.asyncio
async def test_unknown_standard_returns_controlled_not_found():
    service = AlliedStandardsGraphService(FakeNeo4jDriver())

    with pytest.raises(AlliedStandardNotFound):
        await service.get_allied("IS 404:2020")


@pytest.mark.asyncio
async def test_duplicate_loader_run_does_not_duplicate_nodes_or_edges():
    driver = FakeNeo4jDriver()
    service = AlliedStandardsGraphService(driver)
    relations = [
        AlliedStandardRelation(
            source_standard_id="IS 123:2020",
            target_standard_id="IS 456:2000",
            relation_type="references",
        )
    ]

    first = await service.load_relations(relations)
    second = await service.load_relations(relations)

    assert first.nodes_merged == 2
    assert first.edges_merged == 1
    assert second.nodes_merged == 2
    assert second.edges_merged == 1
    assert len(driver.nodes) == 2
    assert len(driver.edges) == 1


def test_allied_endpoint_degrades_gracefully_when_neo4j_unavailable():
    driver = FakeNeo4jDriver()
    driver.unavailable = True
    app = create_app(_test_settings())
    app.state.neo4j = FakeNeo4jClientService(driver)

    with TestClient(app) as client:
        response = client.get("/api/v1/standards/IS%20123%3A2020/allied")

    assert response.status_code == 200
    assert response.json()["allied_standards"] == []
    assert response.json()["warnings"]


def test_allied_endpoint_unknown_standard_returns_not_found():
    app = create_app(_test_settings())
    app.state.neo4j = FakeNeo4jClientService(FakeNeo4jDriver())

    with TestClient(app) as client:
        response = client.get("/api/v1/standards/IS%20404%3A2020/allied")

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "STANDARD_NOT_FOUND"


def _test_settings() -> Settings:
    return Settings(embedding_warmup_on_startup=False)
