from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse

from app.schemas.search import ProductAwareSearchRequest


def register_docs_routes(app: FastAPI) -> None:
    @app.get("/docs", include_in_schema=False)
    async def standardwise_docs() -> HTMLResponse:
        return HTMLResponse(_docs_html(app))

    @app.get("/swagger", include_in_schema=False)
    async def swagger_docs() -> HTMLResponse:
        return get_swagger_ui_html(
            openapi_url=app.openapi_url or "/openapi.json",
            title=f"{app.title} - Swagger UI",
        )


def _schema_for_model(model: type[Any]) -> dict[str, Any]:
    if hasattr(model, "model_json_schema"):
        return model.model_json_schema()
    return model.schema()


def _docs_html(app: FastAPI) -> str:
    schema = _schema_for_model(ProductAwareSearchRequest)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{app.title} - StandardWise Test Console</title>
  <style>
    :root {{
      color-scheme: light;
      --border: #d7dee8;
      --ink: #172033;
      --muted: #607086;
      --soft: #f5f7fb;
      --accent: #185adb;
      --accent-dark: #1249b1;
      --ok: #0d7a48;
      --warn: #9a5b00;
      --bad: #b42318;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family:
        Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: #ffffff;
    }}
    .page {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 24px;
    }}
    .console {{
      border: 1px solid var(--border);
      border-radius: 8px;
      background: #fff;
      box-shadow: 0 8px 30px rgba(23, 32, 51, 0.08);
      overflow: hidden;
    }}
    .console-header {{
      padding: 18px 20px;
      border-bottom: 1px solid var(--border);
      background: linear-gradient(180deg, #ffffff, #f8fafc);
    }}
    .console-header h1 {{
      margin: 0;
      font-size: 22px;
      letter-spacing: 0;
    }}
    .console-header p {{
      margin: 6px 0 0;
      color: var(--muted);
      font-size: 14px;
    }}
    .console-body {{
      display: grid;
      grid-template-columns: minmax(280px, 420px) 1fr;
      gap: 20px;
      padding: 20px;
    }}
    form {{
      display: grid;
      gap: 14px;
      align-content: start;
    }}
    .field {{
      display: grid;
      gap: 6px;
    }}
    label {{
      font-weight: 650;
      font-size: 14px;
    }}
    .required {{
      color: var(--bad);
      margin-left: 3px;
    }}
    .optional {{
      color: var(--muted);
      font-weight: 500;
      margin-left: 4px;
    }}
    input, textarea, select {{
      width: 100%;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px 11px;
      font: inherit;
      color: var(--ink);
      background: #fff;
    }}
    textarea {{
      min-height: 112px;
      resize: vertical;
    }}
    input[type="checkbox"] {{
      width: auto;
      margin-right: 8px;
    }}
    .hint {{
      color: var(--muted);
      font-size: 12px;
    }}
    .actions {{
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }}
    button {{
      border: 0;
      border-radius: 8px;
      padding: 10px 14px;
      font-weight: 700;
      cursor: pointer;
    }}
    .primary {{
      color: #fff;
      background: var(--accent);
    }}
    .primary:hover {{ background: var(--accent-dark); }}
    .secondary {{
      color: var(--ink);
      background: var(--soft);
      border: 1px solid var(--border);
    }}
    .panel-stack {{
      display: grid;
      gap: 14px;
      min-width: 0;
    }}
    .meta {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      min-height: 28px;
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 13px;
      font-weight: 700;
      background: var(--soft);
      color: var(--muted);
    }}
    .badge.ok {{ color: var(--ok); background: #eaf7f0; }}
    .badge.warn {{ color: var(--warn); background: #fff6df; }}
    .badge.bad {{ color: var(--bad); background: #fff0ee; }}
    details {{
      border: 1px solid var(--border);
      border-radius: 8px;
      background: #fff;
    }}
    summary {{
      cursor: pointer;
      padding: 11px 13px;
      font-weight: 700;
    }}
    pre {{
      margin: 0;
      padding: 13px;
      overflow: auto;
      border-top: 1px solid var(--border);
      background: #0f172a;
      color: #dbeafe;
      font-size: 13px;
      line-height: 1.45;
    }}
    .results {{
      display: grid;
      gap: 10px;
    }}
    .result-card {{
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 13px;
      background: #fff;
    }}
    .rank-line {{
      display: flex;
      gap: 10px;
      align-items: baseline;
      flex-wrap: wrap;
    }}
    .rank {{
      font-weight: 800;
      color: var(--accent);
    }}
    .code {{
      font-weight: 800;
    }}
    .title {{
      margin-top: 5px;
      font-size: 15px;
    }}
    .candidate-meta {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 13px;
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }}
    .message {{
      border-radius: 8px;
      padding: 12px;
      background: var(--soft);
      color: var(--muted);
    }}
    .message.error {{
      color: var(--bad);
      background: #fff0ee;
      border: 1px solid #ffd0ca;
    }}
    .swagger-frame {{
      width: 100%;
      height: 760px;
      border: 1px solid var(--border);
      border-radius: 8px;
      margin-top: 24px;
      background: #fff;
    }}
    .swagger-title {{
      margin: 28px 0 10px;
      font-size: 20px;
    }}
    @media (max-width: 860px) {{
      .page {{ padding: 14px; }}
      .console-body {{ grid-template-columns: 1fr; }}
      .swagger-frame {{ height: 620px; }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="console" aria-labelledby="console-title">
      <header class="console-header">
        <h1 id="console-title">StandardWise Test Console</h1>
        <p>Builds JSON for <code>POST /api/v1/search/product-aware</code>.</p>
      </header>
      <div class="console-body">
        <form id="test-console-form">
          <div class="field">
            <label for="preset">Test Example <span class="optional">optional</span></label>
            <select id="preset" name="preset">
              <option value="">Choose an example</option>
            </select>
          </div>
          <div id="schema-fields"></div>
          <div class="actions">
            <button class="primary" type="submit">Search Standards</button>
            <button class="secondary" type="button" id="reset-form">Reset</button>
          </div>
        </form>
        <section class="panel-stack" aria-live="polite">
          <div class="meta">
            <span class="badge" id="status-badge">Idle</span>
            <span class="badge" id="latency-badge">Latency: -</span>
          </div>
          <details open>
            <summary>Generated JSON Request</summary>
            <pre id="request-preview">{{}}</pre>
          </details>
          <div id="readable-results" class="results">
            <div class="message">Run a search to see ranked standards.</div>
          </div>
          <details>
            <summary>Raw JSON Response</summary>
            <pre id="raw-response">{{}}</pre>
          </details>
        </section>
      </div>
    </section>

    <h2 class="swagger-title">Swagger UI</h2>
    <iframe class="swagger-frame" title="Swagger UI" src="/swagger"></iframe>
  </main>

  <script>
    const endpoint = "/api/v1/search/product-aware";
    const requestSchema = {json.dumps(schema)};
    const presets = [
      {{
        label: "Valve - reverse flow",
        values: {{
          product: "valve",
          description: "required to prevent reverse flow in a water pipeline",
          limit: 5
        }}
      }},
      {{
        label: "Cement - decorative work",
        values: {{
          product: "cement",
          description: "for decorative and architectural finishing work",
          limit: 5
        }}
      }},
      {{
        label: "PPC - fly ash",
        values: {{
          product: "Portland pozzolana cement",
          description: "manufactured using fly ash",
          limit: 5
        }}
      }},
      {{
        label: "HDPE sewage pipe",
        values: {{
          product: "pipe",
          description: "HDPE pipe for municipal sewage conveyance",
          limit: 5
        }}
      }},
      {{
        label: "Ambiguous valve",
        values: {{
          product: "valve",
          description: "for water supply",
          limit: 5
        }}
      }}
    ];

    const form = document.getElementById("test-console-form");
    const fieldsRoot = document.getElementById("schema-fields");
    const presetSelect = document.getElementById("preset");
    const requestPreview = document.getElementById("request-preview");
    const rawResponse = document.getElementById("raw-response");
    const readableResults = document.getElementById("readable-results");
    const statusBadge = document.getElementById("status-badge");
    const latencyBadge = document.getElementById("latency-badge");
    const requiredFields = new Set(requestSchema.required || []);
    const propertyEntries = Object.entries(requestSchema.properties || {{}});

    function titleize(name) {{
      return name
        .replace(/_/g, " ")
        .replace(/\\b\\w/g, (match) => match.toUpperCase());
    }}

    function inputTypeFor(name, schema) {{
      if (schema.enum) return "select";
      if (schema.type === "boolean") return "checkbox";
      if (schema.type === "integer" || schema.type === "number") return "number";
      if (schema.type === "array") return "array";
      if (name.toLowerCase().includes("description") || (schema.maxLength || 0) > 1000) {{
        return "textarea";
      }}
      return "text";
    }}

    function fieldDefault(name, schema) {{
      if (schema.default !== undefined) return schema.default;
      if (schema.type === "integer" || schema.type === "number") return "";
      if (schema.type === "boolean") return false;
      return "";
    }}

    function renderFields() {{
      fieldsRoot.innerHTML = "";
      for (const [name, schema] of propertyEntries) {{
        const type = inputTypeFor(name, schema);
        const field = document.createElement("div");
        field.className = "field";
        const label = document.createElement("label");
        label.htmlFor = `field-${{name}}`;
        label.textContent = titleize(name);
        if (requiredFields.has(name)) {{
          const marker = document.createElement("span");
          marker.className = "required";
          marker.textContent = "*";
          label.appendChild(marker);
        }} else {{
          const marker = document.createElement("span");
          marker.className = "optional";
          marker.textContent = "optional";
          label.appendChild(marker);
        }}
        field.appendChild(label);

        let control;
        if (type === "textarea") {{
          control = document.createElement("textarea");
        }} else if (type === "select") {{
          control = document.createElement("select");
          for (const optionValue of schema.enum) {{
            const option = document.createElement("option");
            option.value = optionValue;
            option.textContent = optionValue;
            control.appendChild(option);
          }}
        }} else {{
          control = document.createElement("input");
          control.type = type === "checkbox" ? "checkbox" : type === "number" ? "number" : "text";
        }}
        control.id = `field-${{name}}`;
        control.name = name;
        control.dataset.schemaType = schema.type || "string";
        control.dataset.controlType = type;
        if (schema.minimum !== undefined) control.min = schema.minimum;
        if (schema.maximum !== undefined) control.max = schema.maximum;
        if (schema.maxLength !== undefined) control.maxLength = schema.maxLength;
        if (requiredFields.has(name)) control.required = true;
        const defaultValue = fieldDefault(name, schema);
        if (type === "checkbox") {{
          control.checked = Boolean(defaultValue);
        }} else {{
          control.value = defaultValue;
        }}
        control.addEventListener("input", updatePreview);
        control.addEventListener("change", updatePreview);
        field.appendChild(control);

        const hintParts = [];
        if (schema.minimum !== undefined || schema.maximum !== undefined) {{
          hintParts.push(`range ${{schema.minimum ?? ""}}-${{schema.maximum ?? ""}}`);
        }}
        if (!requiredFields.has(name)) hintParts.push("omitted when empty");
        if (hintParts.length) {{
          const hint = document.createElement("div");
          hint.className = "hint";
          hint.textContent = hintParts.join("; ");
          field.appendChild(hint);
        }}
        fieldsRoot.appendChild(field);
      }}
    }}

    function renderPresets() {{
      for (const preset of presets) {{
        const option = document.createElement("option");
        option.value = preset.label;
        option.textContent = preset.label;
        presetSelect.appendChild(option);
      }}
    }}

    function collectPayload() {{
      const payload = {{}};
      for (const [name, schema] of propertyEntries) {{
        const control = document.querySelector(`[name="${{name}}"]`);
        if (!control) continue;
        const controlType = control.dataset.controlType;
        const schemaType = control.dataset.schemaType;
        let value;
        if (controlType === "checkbox") {{
          value = control.checked;
        }} else if (controlType === "number") {{
          if (control.value === "") {{
            if (requiredFields.has(name)) value = Number(schema.default ?? 0);
            else continue;
          }} else {{
            value = schemaType === "integer" ? parseInt(control.value, 10) : Number(control.value);
          }}
        }} else if (controlType === "array") {{
          if (!control.value.trim()) continue;
          value = control.value.split(",").map((item) => item.trim()).filter(Boolean);
        }} else {{
          if (!control.value.trim() && !requiredFields.has(name)) continue;
          value = control.value;
        }}
        payload[name] = value;
      }}
      return payload;
    }}

    function updatePreview() {{
      requestPreview.textContent = JSON.stringify(collectPayload(), null, 2);
    }}

    function setStatus(status, latencyMs, ok) {{
      statusBadge.textContent = status;
      statusBadge.className = `badge ${{ok === true ? "ok" : ok === false ? "bad" : "warn"}}`;
      latencyBadge.textContent = latencyMs == null ? "Latency: -" : `Latency: ${{latencyMs}} ms`;
    }}

    function formatNumber(value) {{
      if (typeof value !== "number") return value ?? "-";
      return Number.isInteger(value) ? String(value) : value.toFixed(4);
    }}

    function renderResults(data) {{
      const candidates = Array.isArray(data?.candidates) ? data.candidates : [];
      if (!candidates.length) {{
        readableResults.innerHTML = `<div class="message">No candidates returned.</div>`;
        return;
      }}
      readableResults.innerHTML = candidates.map((candidate) => `
        <article class="result-card">
          <div class="rank-line">
            <span class="rank">#${{candidate.rank ?? "-"}}</span>
            <span class="code">${{candidate.standard_code ?? "Unknown standard"}}</span>
          </div>
          <div class="title">${{candidate.title ?? ""}}</div>
          <div class="candidate-meta">
            <span>Score: ${{formatNumber(candidate.reranker_score)}}</span>
            <span>Product: ${{candidate.canonical_product ?? "-"}}</span>
            <span>Family: ${{candidate.family ?? "-"}}</span>
            <span>Compatibility: ${{candidate.product_compatibility ?? "-"}}</span>
          </div>
        </article>
      `).join("");
    }}

    function readableValidationError(data) {{
      const detail = data?.detail;
      if (Array.isArray(detail)) {{
        return detail.map((item) => {{
          const location = Array.isArray(item.loc)
            ? item.loc.filter((part) => part !== "body").join(".")
            : "";
          return `${{location || "Request"}}: ${{item.msg || "Invalid value"}}`;
        }}).join("\\n");
      }}
      if (detail?.message) return detail.message;
      if (typeof detail === "string") return detail;
      return "The request failed. See raw JSON response for details.";
    }}

    async function submitSearch(event) {{
      event.preventDefault();
      const payload = collectPayload();
      updatePreview();
      setStatus("Sending", null, null);
      readableResults.innerHTML = `<div class="message">Searching...</div>`;
      const startedAt = performance.now();
      try {{
        const response = await fetch(endpoint, {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify(payload)
        }});
        const latency = Math.round(performance.now() - startedAt);
        const text = await response.text();
        let data;
        try {{
          data = text ? JSON.parse(text) : {{}};
        }} catch {{
          data = {{ response: text }};
        }}
        rawResponse.textContent = JSON.stringify(data, null, 2);
        setStatus(`HTTP ${{response.status}}`, latency, response.ok);
        if (response.ok) {{
          renderResults(data);
        }} else {{
          const message = readableValidationError(data);
          readableResults.innerHTML = `<div class="message error">${{message}}</div>`;
        }}
      }} catch (error) {{
        const latency = Math.round(performance.now() - startedAt);
        setStatus("Network error", latency, false);
        const message = error instanceof Error ? error.message : String(error);
        readableResults.innerHTML = `<div class="message error">${{message}}</div>`;
        rawResponse.textContent = JSON.stringify({{ error: message }}, null, 2);
      }}
    }}

    presetSelect.addEventListener("change", () => {{
      const preset = presets.find((item) => item.label === presetSelect.value);
      if (!preset) return;
      for (const [name, value] of Object.entries(preset.values)) {{
        const control = document.querySelector(`[name="${{name}}"]`);
        if (!control) continue;
        if (control.type === "checkbox") control.checked = Boolean(value);
        else control.value = value;
      }}
      updatePreview();
    }});

    document.getElementById("reset-form").addEventListener("click", () => {{
      form.reset();
      for (const [name, schema] of propertyEntries) {{
        const control = document.querySelector(`[name="${{name}}"]`);
        if (!control) continue;
        const defaultValue = fieldDefault(name, schema);
        if (control.type === "checkbox") control.checked = Boolean(defaultValue);
        else control.value = defaultValue;
      }}
      setStatus("Idle", null, null);
      readableResults.innerHTML =
        `<div class="message">Run a search to see ranked standards.</div>`;
      rawResponse.textContent = "{{}}";
      updatePreview();
    }});

    form.addEventListener("submit", submitSearch);
    renderFields();
    renderPresets();
    updatePreview();
  </script>
</body>
</html>"""
