from fastapi.testclient import TestClient

from app.main import create_app


def test_custom_docs_page_contains_standardwise_console() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/docs")

    assert response.status_code == 200
    assert "StandardWise Test Console" in response.text
    assert "test-console-form" in response.text
    assert "schema-fields" in response.text
    assert "/api/v1/search/product-aware" in response.text
    assert "Generated JSON Request" in response.text
    assert "Raw JSON Response" in response.text
    assert "Valve - reverse flow" in response.text


def test_swagger_ui_remains_available_separately() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/swagger")

    assert response.status_code == 200
    assert "Swagger UI" in response.text
    assert "/openapi.json" in response.text


def test_openapi_json_remains_accessible_and_product_aware_uses_json_body() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/openapi.json")

    assert response.status_code == 200
    openapi = response.json()
    operation = openapi["paths"]["/api/v1/search/product-aware"]["post"]
    content = operation["requestBody"]["content"]

    assert "application/json" in content
    schema_ref = content["application/json"]["schema"]["$ref"]
    schema_name = schema_ref.rsplit("/", 1)[-1]
    request_schema = openapi["components"]["schemas"][schema_name]

    assert request_schema["required"] == ["product", "description"]
    assert set(request_schema["properties"]) == {"product", "description", "limit"}
    assert request_schema["properties"]["limit"]["default"] == 5


def test_docs_page_embeds_actual_product_aware_schema_fields() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/docs")

    assert '"product"' in response.text
    assert '"description"' in response.text
    assert '"limit"' in response.text
    assert '"minimum": 1' in response.text
    assert '"maximum": 30' in response.text
