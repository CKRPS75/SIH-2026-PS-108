from __future__ import annotations

# ruff: noqa: E501
import json
from typing import Any

from fastapi import FastAPI
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse

from app.schemas.search import ProductAwareSearchRequest
from app.schemas.tenders import TenderTextAnalysisRequest


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
    tender_schema = _schema_for_model(TenderTextAnalysisRequest)
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
      width: 95vw;
      max-width: 1660px;
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
    .tender-console-body {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 18px;
      padding: 20px;
    }}
    .tabs {{
      display: flex;
      gap: 8px;
      padding: 12px 20px 0;
      border-bottom: 1px solid var(--border);
      background: #fff;
    }}
    .tab {{
      border-radius: 8px 8px 0 0;
      border: 1px solid var(--border);
      border-bottom: 0;
      background: var(--soft);
      color: var(--ink);
    }}
    .tab.active {{
      background: #fff;
      color: var(--accent);
    }}
    .tab-panel[hidden] {{
      display: none;
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
    #tender-text {{
      min-height: 360px;
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
    .result-card.primary-result {{
      border-color: #9db8ff;
      background: #f8fbff;
    }}
    .item-card {{
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px;
      background: #fff;
      display: grid;
      gap: 14px;
    }}
    .item-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 12px;
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
    }}
    .summary-cell {{
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px;
      background: #fff;
    }}
    .summary-cell span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }}
    .summary-cell strong {{
      display: block;
      margin-top: 4px;
      font-size: 17px;
    }}
    .section-title {{
      margin: 0;
      font-size: 16px;
    }}
    .subsection-title {{
      margin: 0 0 8px;
      font-size: 14px;
      color: var(--muted);
      text-transform: uppercase;
    }}
    .fact-list {{
      display: grid;
      gap: 8px;
      margin: 0;
    }}
    .fact {{
      border-left: 3px solid #c8d6ee;
      padding-left: 9px;
    }}
    .fact-label {{
      font-weight: 750;
    }}
    .fact-evidence {{
      color: var(--muted);
      font-size: 13px;
      margin-top: 2px;
    }}
    .standard-code {{
      font-size: 24px;
      font-weight: 850;
      color: var(--accent);
    }}
    .standard-title {{
      margin-top: 4px;
      font-weight: 700;
      line-height: 1.35;
    }}
    .muted {{
      color: var(--muted);
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
        <p>Builds JSON for <code>POST /api/v1/search/product-aware</code> and <code>POST /api/v1/tenders/analyze-text</code>.</p>
      </header>
      <nav class="tabs" aria-label="StandardWise console sections">
        <button class="tab active" type="button" data-tab="standard-panel">Standard Search</button>
        <button class="tab" type="button" data-tab="tender-panel">Tender Analyzer</button>
      </nav>
      <div id="standard-panel" class="tab-panel console-body">
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
      <div id="tender-panel" class="tab-panel tender-console-body" hidden>
        <form id="tender-console-form">
          <div class="field">
            <label for="tender-preset">Tender Preset <span class="optional">optional</span></label>
            <select id="tender-preset" name="tender-preset">
              <option value="">Choose a tender</option>
              <option value="upvc">UPVC Drainage Mock Tender</option>
              <option value="construction">3-Item Construction Tender</option>
            </select>
          </div>
          <div class="field">
            <label for="tender-text">Tender Text <span class="required">*</span></label>
            <textarea id="tender-text" name="text" required></textarea>
          </div>
          <div class="field">
            <label for="tender-limit">Limit <span class="optional">optional</span></label>
            <input id="tender-limit" name="limit" type="number" min="1" max="30" value="5" />
            <div class="hint">range 1-30; omitted when empty</div>
          </div>
          <div class="actions">
            <button class="primary" type="submit">Analyze Tender</button>
            <button class="secondary" type="button" id="reset-tender-form">Reset</button>
          </div>
        </form>
        <section class="panel-stack" aria-live="polite">
          <div class="meta">
            <span class="badge" id="tender-status-badge">Idle</span>
            <span class="badge" id="tender-latency-badge">Latency: -</span>
          </div>
          <details open>
            <summary>Generated JSON Request</summary>
            <pre id="tender-request-preview">{{}}</pre>
          </details>
          <div id="tender-readable-results" class="results">
            <div class="message">Analyze tender text to see extracted items and ranked standards.</div>
          </div>
          <details>
            <summary>Raw JSON Response</summary>
            <pre id="tender-raw-response">{{}}</pre>
          </details>
        </section>
      </div>
    </section>

    <h2 class="swagger-title">Swagger UI</h2>
    <iframe class="swagger-frame" title="Swagger UI" src="/swagger"></iframe>
  </main>

  <script>
    const endpoint = "/api/v1/search/product-aware";
    const tenderEndpoint = "/api/v1/tenders/analyze-text";
    const requestSchema = {json.dumps(schema)};
    const tenderRequestSchema = {json.dumps(tender_schema)};
    const upvcTenderText = `Tender Item No. 07 — Supply and Installation of UPVC Drainage Pipes

Supply, delivery, installation, jointing and testing of unplasticized polyvinyl chloride (UPVC) pipes for soil, waste and rainwater drainage applications in residential and commercial buildings. The pipes shall be suitable for conveying domestic wastewater, soil discharge and rainwater from internal and external building drainage systems.

The pipes shall be of 110 mm nominal diameter unless otherwise specified in the Bill of Quantities. The material shall be rigid UPVC, resistant to normal domestic wastewater, moisture and corrosion. Pipes shall have smooth internal surfaces and shall be suitable for gravity drainage applications.

The scope of work shall include cutting, laying, jointing, fixing and connecting the pipes using compatible UPVC fittings, bends, tees, couplers and other accessories required for completion of the drainage system. All joints shall be watertight and shall be made using suitable manufacturer-recommended jointing methods.

The contractor shall provide all necessary clamps, supports, fittings and accessories required for proper installation. The completed piping system shall be checked for leakage and satisfactory flow before acceptance.

The supplied material shall be new, free from manufacturing defects and suitable for use in building soil, waste and rainwater disposal systems.`;
    const constructionTenderText = `Item 1: Supply and lay internal clay flooring tiles for building floor finish. Tiles shall be clay flooring tiles suitable for interior pedestrian use.

Item 2: Supply and install UPVC pipes for building soil, waste and rainwater drainage systems including internal and external drainage connections.

Item 3: Supply HDPE pipes for underground municipal sewage pipeline works suitable for conveyance of sewage in buried sewer networks.`;
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
    const tenderForm = document.getElementById("tender-console-form");
    const tenderPresetSelect = document.getElementById("tender-preset");
    const tenderText = document.getElementById("tender-text");
    const tenderLimit = document.getElementById("tender-limit");
    const tenderRequestPreview = document.getElementById("tender-request-preview");
    const tenderRawResponse = document.getElementById("tender-raw-response");
    const tenderReadableResults = document.getElementById("tender-readable-results");
    const tenderStatusBadge = document.getElementById("tender-status-badge");
    const tenderLatencyBadge = document.getElementById("tender-latency-badge");

    document.querySelectorAll(".tab").forEach((tab) => {{
      tab.addEventListener("click", () => {{
        document.querySelectorAll(".tab").forEach((item) => item.classList.remove("active"));
        document.querySelectorAll(".tab-panel").forEach((panel) => panel.hidden = true);
        tab.classList.add("active");
        document.getElementById(tab.dataset.tab).hidden = false;
      }});
    }});

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

    function collectTenderPayload() {{
      const payload = {{ text: tenderText.value, limit: 5 }};
      if (tenderLimit.value) payload.limit = parseInt(tenderLimit.value, 10);
      return payload;
    }}

    function updateTenderPreview() {{
      tenderRequestPreview.textContent = JSON.stringify(collectTenderPayload(), null, 2);
    }}

    function setTenderStatus(status, latencyMs, ok) {{
      tenderStatusBadge.textContent = status;
      tenderStatusBadge.className = `badge ${{ok === true ? "ok" : ok === false ? "bad" : "warn"}}`;
      tenderLatencyBadge.textContent = latencyMs == null ? "Latency: -" : `Latency: ${{latencyMs}} ms`;
    }}

    function clearNode(node) {{
      while (node.firstChild) node.removeChild(node.firstChild);
    }}

    function node(tag, className, text) {{
      const element = document.createElement(tag);
      if (className) element.className = className;
      if (text !== undefined && text !== null) element.textContent = text;
      return element;
    }}

    function appendSummaryCell(parent, label, value) {{
      const cell = node("div", "summary-cell");
      cell.appendChild(node("span", "", label));
      cell.appendChild(node("strong", "", value ?? "-"));
      parent.appendChild(cell);
    }}

    function appendMeta(parent, label, value) {{
      if (value === undefined || value === null || value === "" || (Array.isArray(value) && !value.length)) return;
      const rendered = Array.isArray(value) ? value.join(", ") : String(value);
      const line = node("span", "", `${{label}}: ${{rendered}}`);
      parent.appendChild(line);
    }}

    function detailsBlock(summary, content) {{
      const wrapper = document.createElement("details");
      wrapper.appendChild(node("summary", "", summary));
      const pre = document.createElement("pre");
      pre.textContent = typeof content === "string" ? content : JSON.stringify(content, null, 2);
      wrapper.appendChild(pre);
      return wrapper;
    }}

    function factsBlock(title, facts) {{
      const section = node("section", "");
      section.appendChild(node("h4", "subsection-title", title));
      const list = node("div", "fact-list");
      const usableFacts = Array.isArray(facts) ? facts.filter((fact) => fact?.value) : [];
      if (!usableFacts.length) {{
        list.appendChild(node("div", "muted", "No source-backed facts found in the dataset text."));
      }} else {{
        for (const fact of usableFacts) {{
          const item = node("div", "fact");
          item.appendChild(node("div", "fact-label", `${{fact.label || "Fact"}}: ${{fact.value}}`));
          list.appendChild(item);
        }}
      }}
      section.appendChild(list);
      return section;
    }}

    function sourceEvidenceBlock(detail) {{
      const facts = [
        ...(detail?.nominal_sizes || []),
        ...(detail?.classification || []),
        ...(detail?.dimensional_requirements || []),
        ...(detail?.performance_requirements || []),
        ...(detail?.workmanship_requirements || []),
        ...(detail?.test_requirements || []),
        ...(detail?.temperature_ranges || []),
        ...(detail?.grades || []),
        ...(detail?.other_requirements || [])
      ].filter((fact) => fact?.evidence);
      const evidence = {{
        overview_source: detail?.source_excerpt || null,
        facts: facts.map((fact) => ({{
          type: fact.type,
          name: fact.name || fact.label,
          value: fact.value,
          qualifier: fact.qualifier,
          unit: fact.unit,
          evidence: fact.evidence
        }}))
      }};
      return detailsBlock("View Source Evidence", evidence);
    }}

    function productSummaryCard(item, index) {{
      const identified = item.identified_product || {{}};
      const req = item.structured_requirement || {{}};
      const card = node("section", "result-card");
      card.appendChild(node("h3", "section-title", "Extracted Procurement Item"));
      const line = node("div", "rank-line");
      line.appendChild(node("span", "rank", `Item ${{item.item_no || index + 1}}`));
      line.appendChild(node("span", "code", identified.display_name || req.product || "Unknown product"));
      card.appendChild(line);
      const grid = node("div", "item-grid");
      appendSummaryCell(grid, "Identified Product", identified.display_name || req.product || "-");
      appendSummaryCell(grid, "Canonical Product", identified.canonical_product || req.normalized_product || "-");
      appendSummaryCell(grid, "Material", (identified.material || req.material || []).join(", ") || "-");
      appendSummaryCell(grid, "Application", (identified.application || req.application || []).join(", ") || "-");
      appendSummaryCell(grid, "Installation", (identified.installation_context || req.installation_context || []).join(", ") || "-");
      appendSummaryCell(grid, "Ambiguity", req.ambiguity ? "Yes" : "No");
      card.appendChild(grid);
      const facts = Array.isArray(identified.important_technical_attributes)
        ? identified.important_technical_attributes.slice(0, 8).map((fact) => ({{
            label: fact.field,
            value: fact.value,
            evidence: fact.evidence
          }}))
        : [];
      card.appendChild(factsBlock("Important Technical Attributes", facts));
      return card;
    }}

    function technicalDetailsSection(detail) {{
      const section = node("section", "result-card");
      section.appendChild(node("h3", "section-title", "Technical Details From Standard"));
      const meta = node("div", "candidate-meta");
      appendMeta(meta, "Product", detail?.product);
      appendMeta(meta, "Subtype", detail?.subtype);
      appendMeta(meta, "Material", detail?.material);
      appendMeta(meta, "Application", detail?.application);
      appendMeta(meta, "Function", detail?.function);
      appendMeta(meta, "Family", detail?.family);
      section.appendChild(meta);
      if (detail?.overview || detail?.scope) {{
        const scope = node("div", "fact");
        scope.appendChild(node("div", "fact-label", "What the Standard Covers"));
        scope.appendChild(node("div", "fact-evidence", detail.overview || detail.scope));
        section.appendChild(scope);
      }}
      section.appendChild(factsBlock("Standard Sizes", detail?.nominal_sizes || detail?.dimensions));
      section.appendChild(factsBlock("Classification", detail?.classification || detail?.grades));
      section.appendChild(factsBlock("Dimensional Requirements", detail?.dimensional_requirements));
      section.appendChild(factsBlock("Temperature Ranges", detail?.temperature_ranges));
      section.appendChild(factsBlock("Performance Requirements", detail?.performance_requirements));
      section.appendChild(factsBlock("Quality / Workmanship", detail?.workmanship_requirements));
      section.appendChild(factsBlock("Test Requirements", detail?.test_requirements));
      section.appendChild(factsBlock("Other Requirements", detail?.other_requirements));
      section.appendChild(sourceEvidenceBlock(detail));
      return section;
    }}

    function recommendationCard(rec, primary) {{
      const card = node("article", primary ? "result-card primary-result" : "result-card");
      card.appendChild(node("h3", "section-title", primary ? "#1 Recommended Standard" : `Alternative #${{rec.rank ?? "-"}}`));
      card.appendChild(node("div", "standard-code", rec.standard_code || "Unknown standard"));
      card.appendChild(node("div", "standard-title", rec.title || ""));
      const meta = node("div", "candidate-meta");
      appendMeta(meta, "Final", formatNumber(rec.final_score));
      appendMeta(meta, "RRF", formatNumber(rec.rrf_score));
      appendMeta(meta, "Compatibility", rec.product_compatibility);
      appendMeta(meta, "Flags", (rec.constraint_flags || []).slice(0, 6));
      card.appendChild(meta);
      if (primary) card.appendChild(technicalDetailsSection(rec.technical_details || {{}}));
      else if (rec.technical_details?.scope) card.appendChild(node("div", "candidate-meta", rec.technical_details.scope));
      return card;
    }}

    function renderTenderResults(data) {{
      const items = Array.isArray(data?.items) ? data.items : [];
      if (!items.length) {{
        clearNode(tenderReadableResults);
        tenderReadableResults.appendChild(node("div", "message", "No structured procurement items returned."));
        return;
      }}
      clearNode(tenderReadableResults);
      const summary = node("section", "result-card");
      summary.appendChild(node("h3", "section-title", "Analysis Summary"));
      const summaryGrid = node("div", "summary-grid");
      appendSummaryCell(summaryGrid, "Items Detected", String(data.item_count ?? items.length));
      appendSummaryCell(summaryGrid, "Status", data.diagnostics?.status || "ok");
      appendSummaryCell(summaryGrid, "Total Latency", `${{formatNumber(data.diagnostics?.total_latency_ms)}} ms`);
      appendSummaryCell(summaryGrid, "Warnings", String((data.diagnostics?.warnings || []).length));
      summary.appendChild(summaryGrid);
      tenderReadableResults.appendChild(summary);

      items.forEach((item, index) => {{
        const payload = item.search_payload || {{}};
        const recs = Array.isArray(item.recommendations) ? item.recommendations : [];
        const itemCard = node("article", "item-card");
        itemCard.appendChild(productSummaryCard(item, index));
        itemCard.appendChild(detailsBlock("Normalized Search Payload", payload));
        if (recs.length) {{
          itemCard.appendChild(recommendationCard(recs[0], true));
          if (recs.length > 1) {{
            const alternatives = node("section", "");
            alternatives.appendChild(node("h3", "section-title", "Alternative Standards"));
            recs.slice(1, 5).forEach((rec) => alternatives.appendChild(recommendationCard(rec, false)));
            itemCard.appendChild(alternatives);
          }}
        }} else {{
          itemCard.appendChild(node("div", "message", "No recommendations returned."));
        }}
        itemCard.appendChild(detailsBlock("Diagnostics", {{
          warnings: item.warnings,
          search_timings_ms: item.search_timings_ms,
          ranking: recs.map((rec) => rec.ranking)
        }}));
        tenderReadableResults.appendChild(itemCard);
      }});

      tenderReadableResults.appendChild(detailsBlock("View Original Tender", data.original_tender_text || ""));
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

    async function submitTender(event) {{
      event.preventDefault();
      const payload = collectTenderPayload();
      updateTenderPreview();
      setTenderStatus("Sending", null, null);
      clearNode(tenderReadableResults);
      tenderReadableResults.appendChild(node("div", "message", "Analyzing tender..."));
      const startedAt = performance.now();
      try {{
        const response = await fetch(tenderEndpoint, {{
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
        tenderRawResponse.textContent = JSON.stringify(data, null, 2);
        setTenderStatus(`HTTP ${{response.status}}`, latency, response.ok);
        if (response.ok) renderTenderResults(data);
        else {{
          clearNode(tenderReadableResults);
          tenderReadableResults.appendChild(node("div", "message error", readableValidationError(data)));
        }}
      }} catch (error) {{
        const latency = Math.round(performance.now() - startedAt);
        const message = error instanceof Error ? error.message : String(error);
        setTenderStatus("Network error", latency, false);
        clearNode(tenderReadableResults);
        tenderReadableResults.appendChild(node("div", "message error", message));
        tenderRawResponse.textContent = JSON.stringify({{ error: message }}, null, 2);
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

    tenderText.addEventListener("input", updateTenderPreview);
    tenderLimit.addEventListener("input", updateTenderPreview);
    tenderPresetSelect.addEventListener("change", () => {{
      if (tenderPresetSelect.value === "upvc") {{
        tenderText.value = upvcTenderText;
        tenderLimit.value = "5";
      }} else if (tenderPresetSelect.value === "construction") {{
        tenderText.value = constructionTenderText;
        tenderLimit.value = "5";
      }}
      updateTenderPreview();
    }});
    document.getElementById("reset-tender-form").addEventListener("click", () => {{
      tenderForm.reset();
      tenderLimit.value = "5";
      setTenderStatus("Idle", null, null);
      tenderReadableResults.innerHTML =
        `<div class="message">Analyze tender text to see extracted items and ranked standards.</div>`;
      tenderRawResponse.textContent = "{{}}";
      updateTenderPreview();
    }});

    form.addEventListener("submit", submitSearch);
    tenderForm.addEventListener("submit", submitTender);
    renderFields();
    renderPresets();
    updatePreview();
    updateTenderPreview();
  </script>
</body>
</html>"""
