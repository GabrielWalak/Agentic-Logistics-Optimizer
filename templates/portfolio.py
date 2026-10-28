"""Portfolio page HTML template, separated from the API entry point."""

from html import escape


def build_home_page(base_url: str, llm_model: str) -> str:
    """Render portfolio-style home page showcasing AI/ML architecture."""
    llm_model_label = escape(llm_model)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Agent Logistics AI | Portfolio</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', -apple-system, sans-serif; background: #0d1117; color: #c9d1d9; line-height: 1.6; }}
        .container {{ max-width: 1100px; margin: 0 auto; padding: 40px 20px; }}
        .header {{ text-align: center; margin-bottom: 50px; }}
        .header h1 {{ font-size: 2.2em; color: #58a6ff; margin-bottom: 8px; font-weight: 600; }}
        .header .subtitle {{ color: #8b949e; font-size: 1.1em; }}
        .badge-row {{ margin-top: 16px; display: flex; gap: 8px; justify-content: center; flex-wrap: wrap; }}
        .badge {{ background: #21262d; border: 1px solid #30363d; padding: 4px 12px; border-radius: 20px; font-size: 0.8em; color: #79c0ff; }}
        .section {{ margin-bottom: 40px; }}
        .section h2 {{ color: #58a6ff; font-size: 1.3em; margin-bottom: 16px; padding-bottom: 8px; border-bottom: 1px solid #21262d; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; }}
        .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; }}
        .card-title {{ color: #58a6ff; font-size: 0.9em; font-weight: 600; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px; }}
        .card-value {{ color: #f0f6fc; font-size: 1.8em; font-weight: 700; }}
        .card-sub {{ color: #8b949e; font-size: 0.85em; margin-top: 4px; }}
        .arch {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 24px; font-family: 'Cascadia Code', 'Fira Code', monospace; font-size: 0.82em; white-space: pre; overflow-x: auto; color: #8b949e; line-height: 1.8; }}
        .arch .highlight {{ color: #58a6ff; }}
        .arch .green {{ color: #3fb950; }}
        .arch .orange {{ color: #d29922; }}
        .arch .pink {{ color: #f778ba; }}
        .tech-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }}
        .tech-item {{ background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 12px 16px; }}
        .tech-item .tech-label {{ color: #8b949e; font-size: 0.75em; text-transform: uppercase; letter-spacing: 0.5px; }}
        .tech-item .tech-value {{ color: #c9d1d9; font-size: 0.9em; margin-top: 2px; }}
        .demo {{ background: #0d1117; border: 1px solid #238636; border-radius: 8px; padding: 20px; }}
        .demo-row {{ display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #21262d; }}
        .demo-row:last-child {{ border-bottom: none; }}
        .demo-label {{ color: #8b949e; }}
        .demo-value {{ color: #f0f6fc; font-weight: 500; }}
        .demo-value.high {{ color: #f85149; }}
        .demo-value.good {{ color: #3fb950; }}
        .grade-bar {{ height: 6px; background: #21262d; border-radius: 3px; margin-top: 6px; overflow: hidden; }}
        .grade-fill {{ height: 100%; border-radius: 3px; }}
        .grade-excellent {{ background: #3fb950; }}
        .grade-good {{ background: #58a6ff; }}
        .links {{ display: flex; gap: 12px; margin-top: 20px; flex-wrap: wrap; }}
        .link-btn {{ background: #21262d; border: 1px solid #30363d; padding: 10px 20px; border-radius: 6px; color: #58a6ff; text-decoration: none; font-size: 0.9em; transition: background 0.2s; }}
        .link-btn:hover {{ background: #30363d; }}
        .footer {{ text-align: center; margin-top: 50px; padding-top: 20px; border-top: 1px solid #21262d; color: #484f58; font-size: 0.85em; }}
        .scenario-btn {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; cursor: pointer; transition: all 0.2s; text-align: center; color: #c9d1d9; }}
        .scenario-btn:hover {{ border-color: #58a6ff; background: #1c2128; }}
        .scenario-btn:disabled {{ opacity: 0.5; cursor: wait; }}
        .scenario-title {{ color: #58a6ff; font-weight: 700; font-size: 1em; margin-bottom: 6px; }}
        .scenario-desc {{ color: #8b949e; font-size: 0.82em; }}
        .loading-bar {{ height: 4px; background: #21262d; border-radius: 2px; overflow: hidden; }}
        .loading-fill {{ height: 100%; background: linear-gradient(90deg, #58a6ff, #3fb950); border-radius: 2px; transition: width 0.5s ease; width: 0%; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Multi-Agent Logistics AI System</h1>
            <p class="subtitle">Enterprise-grade AI orchestration for real-time logistics decision-making</p>
            <div class="badge-row">
                <span class="badge">Python 3.11</span>
                <span class="badge">FastAPI</span>
                <span class="badge">{llm_model_label}</span>
                <span class="badge">Multi-Agent</span>
                <span class="badge">RAG</span>
                <span class="badge">PostgreSQL</span>
                <span class="badge">Redis</span>
                <span class="badge">Docker</span>
            </div>
        </div>

        <div class="section">
            <h2>System Architecture</h2>
            <div class="arch"><span class="highlight">Delivery Scenario Input</span> (distance, weight, time, payment)
        |
        v
<span class="green">[Agent 1: Risk Assessment]</span> ──────────────────── Score: 0-100
        |                                          Parallel
        ├── <span class="orange">[Agent 2: Carrier Optimization]</span> ──── ROI Analysis
        |                                          Execution
        └── <span class="orange">[Agent 3: Recovery Strategy]</span> ────── Voucher Logic
                        |
                        v
        <span class="pink">[Agent 4: Decision Orchestrator]</span> ──── Integration
                        |
                        v
        <span class="highlight">[Response Grader]</span> ──── Behavioral Validation (0-100)
                        |
                        v
        <span class="green">Final Decision + Confidence Score</span></div>
        </div>

        <div class="section">
            <h2>Technology Stack</h2>
            <div class="tech-grid">
                <div class="tech-item"><div class="tech-label">LLM / AI</div><div class="tech-value">{llm_model_label} via Gemini OpenAI-compatible API</div></div>
                <div class="tech-item"><div class="tech-label">Framework</div><div class="tech-value">FastAPI + Pydantic V2</div></div>
                <div class="tech-item"><div class="tech-label">Architecture</div><div class="tech-value">Multi-Agent Orchestration (4 agents)</div></div>
                <div class="tech-item"><div class="tech-label">Knowledge Base</div><div class="tech-value">RAG + ChromaDB Vector Store</div></div>
                <div class="tech-item"><div class="tech-label">Database</div><div class="tech-value">PostgreSQL + SQLModel + async SQLAlchemy</div></div>
                <div class="tech-item"><div class="tech-label">Cache</div><div class="tech-value">Redis LLM response cache</div></div>
                <div class="tech-item"><div class="tech-label">Infrastructure</div><div class="tech-value">Azure VM + Docker Compose + GitHub Actions</div></div>
                <div class="tech-item"><div class="tech-label">Observability</div><div class="tech-value">Structured JSON logs + optional LangSmith tracing</div></div>
                <div class="tech-item"><div class="tech-label">Evaluation</div><div class="tech-value">Deterministic behavioral grading</div></div>
                <div class="tech-item"><div class="tech-label">Concurrency</div><div class="tech-value">ThreadPoolExecutor for agents 2–3</div></div>
                <div class="tech-item"><div class="tech-label">Security</div><div class="tech-value">API key + portfolio Basic Auth</div></div>
                <div class="tech-item"><div class="tech-label">Prompt Engineering</div><div class="tech-value">Grounded prompts + Pydantic structured outputs</div></div>
            </div>
        </div>

        <div class="section">
            <h2>Latest Analysis Result</h2>
            <p style="color: #8b949e; margin-bottom: 12px; font-size: 0.9em;">This panel is updated with the actual API response after every scenario run.</p>
            <div id="live-result">
                <div class="demo"><span class="demo-label">Select a scenario below to run ML, RAG and the four-agent workflow.</span></div>
            </div>
        </div>

        <div class="section">
            <h2>Run Live AI Analysis</h2>
            <p style="color: #8b949e; margin-bottom: 16px; font-size: 0.9em;">Click a scenario to trigger the real ML, RAG and multi-agent workflow. Runtime depends on the provider and Redis cache.</p>
            <div class="grid" style="grid-template-columns: repeat(3, 1fr);">
                <button class="scenario-btn" onclick="runScenario('high')"><div class="scenario-title">LONG-HAUL</div><div class="scenario-desc">2800km · 4500g · weekend</div></button>
                <button class="scenario-btn" onclick="runScenario('moderate')"><div class="scenario-title">REGIONAL</div><div class="scenario-desc">650km · 2000g · weekday</div></button>
                <button class="scenario-btn" onclick="runScenario('low')"><div class="scenario-title">LOCAL</div><div class="scenario-desc">45km · 300g · on time</div></button>
            </div>
            <div id="live-status" style="margin-top: 16px; display: none;">
                <div class="loading-bar"><div class="loading-fill" id="loading-fill"></div></div>
                <p id="status-text" style="color: #8b949e; font-size: 0.85em; margin-top: 8px;"></p>
            </div>
        </div>

        <div class="section">
            <h2>AI Response Grading Framework</h2>
            <p style="color: #8b949e; margin-bottom: 16px; font-size: 0.9em;">Behavioral validation — not just JSON format, but logical correctness</p>
            <div class="grid">
                <div class="card"><div class="card-title">Score-Level Alignment</div><div class="card-sub">Validates risk_score matches risk_level range</div><div class="grade-bar"><div class="grade-fill grade-excellent" style="width: 25%;"></div></div><div class="card-sub" style="margin-top: 4px;">25 points</div></div>
                <div class="card"><div class="card-title">Factor Specificity</div><div class="card-sub">Requires measurable factors with units (km, kg, days)</div><div class="grade-bar"><div class="grade-fill grade-excellent" style="width: 20%;"></div></div><div class="card-sub" style="margin-top: 4px;">20 points</div></div>
                <div class="card"><div class="card-title">Logic Consistency</div><div class="card-sub">upgrade=true → cost&gt;0, discount matches voucher</div><div class="grade-bar"><div class="grade-fill grade-good" style="width: 20%;"></div></div><div class="card-sub" style="margin-top: 4px;">20 points</div></div>
                <div class="card"><div class="card-title">Financial Grounding</div><div class="card-sub">Uses verified quote costs and accepts an explicit data limitation</div><div class="grade-bar"><div class="grade-fill grade-good" style="width: 25%;"></div></div><div class="card-sub" style="margin-top: 4px;">25 points</div></div>
            </div>
        </div>

        <div class="section">
            <h2>Performance Metrics</h2>
            <p style="color: #8b949e; margin-bottom: 12px; font-size: 0.9em;">Updated from the latest live API response</p>
            <div class="grid">
                <div class="card"><div class="card-title">Processing Time</div><div id="metric-processing" class="card-value">—</div><div class="card-sub">Includes ML, RAG and agent orchestration</div></div>
                <div class="card"><div class="card-title">Specialist Scores</div><div id="metric-specialists" class="card-value" style="font-size: 1.15em;">—</div><div class="card-sub">Risk · Carrier · Recovery</div></div>
                <div class="card"><div class="card-title">Combined Quality Score</div><div id="metric-grading" class="card-value" style="color: #3fb950;">—</div><div id="metric-grading-formula" class="card-sub">Arithmetic mean of the 3 specialist scores</div></div>
            </div>
        </div>

        <div class="section">
            <h2>ML Model &amp; Data Context</h2>
            <div class="card" style="border-left: 3px solid #d29922;">
                <div class="card-title" style="color: #d29922;">About the Prediction Model</div>
                <p style="color: #c9d1d9; font-size: 0.9em; line-height: 1.7; margin-top: 8px;">The delivery time prediction model is trained on the <strong style="color: #f0f6fc;">Brazilian E-Commerce (Olist) dataset</strong> — ~100k orders from 2016-2018. The model provides estimated delivery times that feed into the multi-agent decision system.</p>
                <p style="color: #8b949e; font-size: 0.85em; line-height: 1.6; margin-top: 10px;"><strong style="color: #d29922;">Known limitations:</strong> Does not account for real-time weather, carrier fleet availability, traffic disruptions, or holiday surges. In production, the system would integrate live carrier APIs and weather data.</p>
            </div>
        </div>

        <div class="section">
            <h2>API Endpoints</h2>
            <div class="links">
                <a href="{base_url}/docs" class="link-btn">Swagger UI</a>
                <a href="{base_url}/redoc" class="link-btn">ReDoc</a>
                <a href="{base_url}/health" class="link-btn">Health Check</a>
                <a href="{base_url}/status" class="link-btn">Service Status</a>
            </div>
        </div>

        <div class="footer"><p>Multi-Agent Logistics AI &middot; FastAPI + Gemini + RAG + PostgreSQL + Redis + Docker</p></div>
    </div>
    <script>
    const DEMO_URL = "{base_url}/demo/analyze";
    function escapeHtml(value) {{
        const node = document.createElement('div');
        node.textContent = String(value ?? '');
        return node.innerHTML;
    }}

    async function runScenario(level) {{
        const btns = document.querySelectorAll('.scenario-btn');
        btns.forEach(b => b.disabled = true);
        const statusDiv = document.getElementById('live-status');
        const resultDiv = document.getElementById('live-result');
        const statusText = document.getElementById('status-text');
        const loadingFill = document.getElementById('loading-fill');
        statusDiv.style.display = 'block';
        resultDiv.style.display = 'none';
        statusText.textContent = 'ML Model predicting delivery time...';
        loadingFill.style.width = '10%';
        let progress = 10;
        const interval = setInterval(() => {{
            progress = Math.min(progress + Math.random() * 8, 90);
            loadingFill.style.width = progress + '%';
            if (progress > 20) statusText.textContent = 'Agent 1: Risk Assessment...';
            if (progress > 45) statusText.textContent = 'Agents 2-3: Carrier + Recovery (parallel)...';
            if (progress > 70) statusText.textContent = 'Agent 4: Decision Integration...';
        }}, 1500);
        try {{
            const resp = await fetch(DEMO_URL, {{ method: 'POST', headers: {{ 'Content-Type': 'application/json' }}, body: JSON.stringify({{ scenario: level }}) }});
            clearInterval(interval);
            loadingFill.style.width = '100%';
            if (!resp.ok) {{ const err = await resp.json(); statusText.textContent = 'Error: ' + (err.detail || resp.statusText); btns.forEach(b => b.disabled = false); return; }}
            const data = await resp.json();
            const elapsedSeconds = (data.processing_time_ms / 1000).toFixed(1);
            const fallbackLabel = data.fallback_used ? ' · deterministic fallback' : '';
            const cacheLabel = data.cache_hit ? ' · Redis response hit' : data.cache_enabled ? ' · Redis cache active' : ' · Redis cache unavailable';
            statusText.textContent = `Completed in ${{elapsedSeconds}}s${{fallbackLabel}}${{cacheLabel}}`;
            const d = data.decision, g = data.grading, ml = data.ml_prediction;
            const riskGrade = Number(g.risk_grading.score);
            const carrierGrade = Number(g.carrier_grading.score);
            const recoveryGrade = Number(g.recovery_grading.score);
            document.getElementById('metric-processing').textContent = `${{elapsedSeconds}}s`;
            document.getElementById('metric-specialists').textContent = `${{riskGrade}} · ${{carrierGrade}} · ${{recoveryGrade}}`;
            document.getElementById('metric-grading').textContent = `${{g.overall_score}}/100`;
            document.getElementById('metric-grading-formula').textContent = `(${{riskGrade}} + ${{carrierGrade}} + ${{recoveryGrade}}) ÷ 3`;
            const riskColor = d.risk_assessment.risk_level === 'HIGH' || d.risk_assessment.risk_level === 'CRITICAL' ? '#f85149' : d.risk_assessment.risk_level === 'MODERATE' ? '#d29922' : '#3fb950';
            const view = {{
                riskLevel: escapeHtml(d.risk_assessment.risk_level),
                riskScore: escapeHtml(d.risk_assessment.risk_score),
                qualityLevel: escapeHtml(g.quality_level),
                gradingScore: escapeHtml(g.overall_score),
                predictedDays: escapeHtml(ml.predicted_days),
                riskFactors: escapeHtml(d.risk_assessment.primary_risk_factors.join(', ')),
                analysis: escapeHtml(d.risk_assessment.analysis),
                carrier: escapeHtml(d.carrier_recommendation.recommended_carrier),
                roi: escapeHtml(d.carrier_recommendation.roi_analysis),
                voucher: escapeHtml(d.recovery_plan.voucher_code || 'None'),
                discount: escapeHtml(d.recovery_plan.discount_percentage),
                summary: escapeHtml(d.executive_summary),
                confidence: escapeHtml(d.confidence_score),
            }};
            resultDiv.innerHTML = `<div class="demo" style="border-color: ${{riskColor}};"><div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;"><span style="color:${{riskColor}};font-weight:700;font-size:1.1em;">${{view.riskLevel}} RISK (${{view.riskScore}}/100)</span><span style="color:#3fb950;font-size:0.9em;">Grading: ${{view.gradingScore}}/100 (${{view.qualityLevel}})</span></div><div class="demo-row" style="background:#1c2128;padding:8px;border-radius:4px;margin-bottom:8px;"><span class="demo-label" style="color:#58a6ff;">ML Prediction (XGBoost)</span><span class="demo-value" style="color:#58a6ff;">${{view.predictedDays}} days ${{ml.ml_model_used ? '✓ model used' : '(fallback)'}}</span></div><div class="demo-row"><span class="demo-label">Risk Factors</span><span class="demo-value">${{view.riskFactors}}</span></div><div class="demo-row"><span class="demo-label">Analysis</span><span class="demo-value" style="font-size:0.83em;max-width:650px;">${{view.analysis}}</span></div><div class="demo-row"><span class="demo-label">Carrier</span><span class="demo-value">${{view.carrier}} ${{d.carrier_recommendation.should_upgrade ? '(upgrade)' : ''}}</span></div><div class="demo-row"><span class="demo-label">ROI</span><span class="demo-value" style="font-size:0.83em;max-width:650px;">${{view.roi}}</span></div><div class="demo-row"><span class="demo-label">Recovery</span><span class="demo-value">${{view.voucher}} (${{view.discount}}% off)</span></div><div class="demo-row" style="flex-direction:column;gap:6px;padding-top:10px;border-top:1px solid #30363d;"><span class="demo-label">Executive Summary</span><span class="demo-value" style="font-size:0.88em;line-height:1.6;">${{view.summary}}</span></div><div style="margin-top:12px;padding-top:10px;border-top:1px solid #21262d;display:flex;gap:20px;font-size:0.8em;color:#8b949e;"><span>Confidence: <strong style="color:#f0f6fc;">${{view.confidence}}/100</strong></span><span>Time: <strong style="color:#f0f6fc;">${{elapsedSeconds}}s</strong></span></div></div>`;
            resultDiv.style.display = 'block';
            resultDiv.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
        }} catch (e) {{ clearInterval(interval); statusText.textContent = 'Network error: ' + e.message; }}
        btns.forEach(b => b.disabled = false);
    }}
    </script>
</body>
</html>"""
    return html
