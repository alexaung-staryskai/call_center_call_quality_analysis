from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "output" / "analysis_results.csv"

app = FastAPI(title="Call Analysis Dashboard")

REQUIRED_COLUMNS = [
    "file_name",
    "product_focus",
    "type of call",
    "sentiment_score",
    "did customer get the answer",
    "next step for customer",
    "call_summary",
]


def normalize_product_name(value: str) -> str:
    text = (value or "").strip()
    if not text or text.lower() == "unknown":
        return "Unknown"

    text = " ".join(text.split())

    suspicious_prefixes = [
        "This Is ",
        "My Name Is ",
        "From You ",
        "Last Month ",
    ]
    if len(text) > 60 or any(text.startswith(prefix) for prefix in suspicious_prefixes):
        return f"Needs Review: {text[:40].strip()}..."

    return text


def classify_call_health(row: pd.Series) -> str:
    type_of_call = str(row.get("type of call", "")).strip().lower()
    sentiment = int(row.get("sentiment_score", 3))
    answered = str(row.get("did customer get the answer", "")).strip().lower() == "yes"
    follow_up = str(row.get("next step for customer", "")).strip().lower() == "follow up call"

    if type_of_call == "support":
        return "bad"
    if sentiment <= 2:
        return "bad"
    if type_of_call == "after sales" and follow_up:
        return "bad"
    if answered and sentiment >= 4:
        return "good"
    if type_of_call == "after sales" and sentiment >= 4 and not follow_up:
        return "good"
    return "neutral"


def build_follow_up_action(row: pd.Series) -> str:
    type_of_call = str(row.get("type of call", "")).strip().lower()
    product = str(row.get("product_focus_clean", "Unknown")).strip()
    summary = str(row.get("call_summary", "")).strip()
    answered = str(row.get("did customer get the answer", "")).strip().lower() == "yes"

    if answered:
        return "No follow-up required."

    if type_of_call == "support":
        return f"Urgent callback for {product}: provide remediation status, ETA, and next technical step."
    if type_of_call == "after sales":
        return f"Follow up on {product}: discuss refund, return, replacement, or complaint resolution."
    if type_of_call == "technical qualification":
        return f"Arrange a qualification follow-up for {product}: answer fit, requirements, and implementation questions."
    if type_of_call == "enquiry":
        return f"Follow up on {product}: answer pricing, availability, shipping, or product detail questions."

    if summary:
        return f"Review this call and prepare a callback: {summary}"
    return "General follow-up call required."


def sentiment_label(score: int) -> str:
    if score <= 2:
        return "Negative"
    if score == 3:
        return "Neutral"
    return "Positive"


def load_analysis_csv(csv_path: Path = CSV_PATH) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}. "
            "Make sure analysis_results.csv exists under ./output/"
        )

    df = pd.read_csv(csv_path, dtype=str).fillna("")

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")

    df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce").fillna(3).astype(int)
    df["type of call"] = df["type of call"].astype(str).str.strip().str.lower()
    df["did customer get the answer"] = df["did customer get the answer"].astype(str).str.strip().str.lower()
    df["next step for customer"] = df["next step for customer"].astype(str).str.strip().str.lower()
    df["product_focus_clean"] = df["product_focus"].apply(normalize_product_name)
    df["call_health"] = df.apply(classify_call_health, axis=1)
    df["needs_follow_up"] = (
        (df["did customer get the answer"] != "yes")
        | (df["next step for customer"] == "follow up call")
    )
    df["follow_up_action"] = df.apply(build_follow_up_action, axis=1)
    df["sentiment_label"] = df["sentiment_score"].apply(sentiment_label)

    return df


def build_dashboard_data(csv_path: Path = CSV_PATH) -> dict[str, Any]:
    df = load_analysis_csv(csv_path)
    known_products = df[df["product_focus_clean"] != "Unknown"].copy()

    metrics = {
        "total_calls": int(len(df)),
        "good_calls": int((df["call_health"] == "good").sum()),
        "bad_calls": int((df["call_health"] == "bad").sum()),
        "neutral_calls": int((df["call_health"] == "neutral").sum()),
        "follow_up_calls": int(df["needs_follow_up"].sum()),
        "answered_calls": int((df["did customer get the answer"] == "yes").sum()),
        "avg_sentiment": round(float(df["sentiment_score"].mean()), 2) if len(df) else 0.0,
    }

    bad_calls = (
        df[df["call_health"] == "bad"]
        .sort_values(by=["sentiment_score", "type of call", "file_name"], ascending=[True, True, True])
        [[
            "file_name",
            "product_focus_clean",
            "type of call",
            "sentiment_score",
            "did customer get the answer",
            "next step for customer",
            "call_summary",
            "follow_up_action",
        ]]
        .rename(columns={"product_focus_clean": "product_focus"})
    )

    good_calls = (
        df[df["call_health"] == "good"]
        .sort_values(by=["sentiment_score", "file_name"], ascending=[False, True])
        [[
            "file_name",
            "product_focus_clean",
            "type of call",
            "sentiment_score",
            "did customer get the answer",
            "next step for customer",
            "call_summary",
        ]]
        .rename(columns={"product_focus_clean": "product_focus"})
    )

    follow_ups = (
        df[df["needs_follow_up"]]
        .sort_values(by=["sentiment_score", "type of call", "file_name"], ascending=[True, True, True])
        [[
            "file_name",
            "product_focus_clean",
            "type of call",
            "sentiment_score",
            "call_summary",
            "follow_up_action",
        ]]
        .rename(columns={"product_focus_clean": "product_focus"})
    )

    interest_products = (
        known_products[known_products["type of call"].isin(["enquiry", "technical qualification"])]
        .groupby("product_focus_clean", dropna=False)
        .agg(
            interest_calls=("file_name", "count"),
            open_follow_ups=("needs_follow_up", "sum"),
            avg_sentiment=("sentiment_score", "mean"),
        )
        .sort_values(by=["interest_calls", "open_follow_ups", "avg_sentiment"], ascending=[False, False, False])
        .reset_index()
        .rename(columns={"product_focus_clean": "product_focus"})
    )
    if not interest_products.empty:
        interest_products["avg_sentiment"] = interest_products["avg_sentiment"].round(2)

    issue_products = (
        known_products[
            (known_products["type of call"].isin(["support", "after sales"]))
            | (known_products["sentiment_score"] <= 2)
            | (known_products["needs_follow_up"])
        ]
        .groupby("product_focus_clean", dropna=False)
        .agg(
            issue_calls=("file_name", "count"),
            support_calls=("type of call", lambda s: int((s == "support").sum())),
            after_sales_calls=("type of call", lambda s: int((s == "after sales").sum())),
            avg_sentiment=("sentiment_score", "mean"),
        )
        .sort_values(by=["issue_calls", "support_calls", "after_sales_calls", "avg_sentiment"], ascending=[False, False, False, True])
        .reset_index()
        .rename(columns={"product_focus_clean": "product_focus"})
    )
    if not issue_products.empty:
        issue_products["avg_sentiment"] = issue_products["avg_sentiment"].round(2)

    praise_products = (
        known_products[
            (known_products["call_health"] == "good")
            | ((known_products["type of call"] == "after sales") & (known_products["sentiment_score"] >= 4))
        ]
        .groupby("product_focus_clean", dropna=False)
        .agg(
            positive_calls=("file_name", "count"),
            avg_sentiment=("sentiment_score", "mean"),
        )
        .sort_values(by=["positive_calls", "avg_sentiment"], ascending=[False, False])
        .reset_index()
        .rename(columns={"product_focus_clean": "product_focus"})
    )
    if not praise_products.empty:
        praise_products["avg_sentiment"] = praise_products["avg_sentiment"].round(2)

    type_product_matrix = pd.pivot_table(
        known_products,
        index="product_focus_clean",
        columns="type of call",
        values="file_name",
        aggfunc="count",
        fill_value=0,
    ).reset_index().rename(columns={"product_focus_clean": "product_focus"})

    call_health_dist = (
        df.groupby("call_health", dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(by="count", ascending=False)
    )

    sentiment_dist = (
        df.groupby("sentiment_label", dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(by="count", ascending=False)
    )

    call_type_dist = (
        df.groupby("type of call", dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(by="count", ascending=False)
    )

    follow_up_by_product = (
        known_products[known_products["needs_follow_up"]]
        .groupby("product_focus_clean", dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(by="count", ascending=False)
        .rename(columns={"product_focus_clean": "product_focus"})
    )

    issue_type_dist = (
        df[df["call_health"] == "bad"]
        .groupby("type of call", dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(by="count", ascending=False)
    )

    return {
        "df": df,
        "metrics": metrics,
        "bad_calls": bad_calls,
        "good_calls": good_calls,
        "follow_ups": follow_ups,
        "interest_products": interest_products,
        "issue_products": issue_products,
        "praise_products": praise_products,
        "type_product_matrix": type_product_matrix,
        "call_health_dist": call_health_dist,
        "sentiment_dist": sentiment_dist,
        "call_type_dist": call_type_dist,
        "follow_up_by_product": follow_up_by_product,
        "issue_type_dist": issue_type_dist,
    }


def metric_card(title: str, value: Any, tone: str = "neutral") -> str:
    return f"""
    <div class="card metric {tone}">
        <div class="metric-title">{escape(str(title))}</div>
        <div class="metric-value">{escape(str(value))}</div>
    </div>
    """


def format_cell(value: Any) -> str:
    return escape("" if value is None else str(value))


def dataframe_to_html(df: pd.DataFrame, empty_message: str = "No rows to display.") -> str:
    if df.empty:
        return f'<div class="empty">{escape(empty_message)}</div>'

    headers = "".join(f"<th>{escape(str(col))}</th>" for col in df.columns)
    rows = []
    for _, row in df.iterrows():
        cells = "".join(f"<td>{format_cell(row[col])}</td>" for col in df.columns)
        rows.append(f"<tr>{cells}</tr>")

    return f"""
    <div class="table-wrap">
        <table>
            <thead><tr>{headers}</tr></thead>
            <tbody>{''.join(rows)}</tbody>
        </table>
    </div>
    """


def bar_chart_html(df: pd.DataFrame, label_col: str, value_col: str, title: str) -> str:
    if df.empty:
        return f"""
        <div class="card">
            <h3>{escape(title)}</h3>
            <div class="empty">No data available.</div>
        </div>
        """

    max_value = max(df[value_col].max(), 1)
    bars = []

    for _, row in df.iterrows():
        label = str(row[label_col])
        value = int(row[value_col])
        width = max(6, int((value / max_value) * 100))
        bars.append(
            f"""
            <div class="bar-row">
                <div class="bar-label">{escape(label)}</div>
                <div class="bar-track">
                    <div class="bar-fill" style="width: {width}%"></div>
                </div>
                <div class="bar-value">{value}</div>
            </div>
            """
        )

    return f"""
    <div class="card">
        <h3>{escape(title)}</h3>
        <div class="bars">
            {''.join(bars)}
        </div>
    </div>
    """


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    try:
        data = build_dashboard_data(CSV_PATH)
    except Exception as exc:
        return f"""
        <html>
        <body style="font-family: Arial, sans-serif; padding: 24px;">
            <h1>Call Analysis Dashboard</h1>
            <p>Unable to load dashboard.</p>
            <pre>{escape(str(exc))}</pre>
        </body>
        </html>
        """

    metrics_html = "".join(
        [
            metric_card("Total Calls", data["metrics"]["total_calls"]),
            metric_card("Good Calls", data["metrics"]["good_calls"], "good"),
            metric_card("Bad Calls", data["metrics"]["bad_calls"], "bad"),
            metric_card("Neutral Calls", data["metrics"]["neutral_calls"]),
            metric_card("Follow-up Calls", data["metrics"]["follow_up_calls"], "warn"),
            metric_card("Average Sentiment", data["metrics"]["avg_sentiment"]),
        ]
    )

    calls_tab = f"""
    <div id="tab-calls" class="tab-panel active">
        <div class="grid metrics">{metrics_html}</div>

        <div class="grid three-col section">
            {bar_chart_html(data["call_health_dist"], "call_health", "count", "Call Health Distribution")}
            {bar_chart_html(data["sentiment_dist"], "sentiment_label", "count", "Sentiment Distribution")}
            {bar_chart_html(data["call_type_dist"], "type of call", "count", "Call Type Distribution")}
        </div>

        <div class="grid two-col section">
            <div class="card">
                <h2>Good Calls</h2>
                <div class="section-note">Calls that look positive and resolved.</div>
                {dataframe_to_html(data["good_calls"], "No good calls found.")}
            </div>
            <div class="card">
                <h2>Bad Calls</h2>
                <div class="section-note">Calls that need immediate attention.</div>
                {dataframe_to_html(data["bad_calls"], "No bad calls found.")}
            </div>
        </div>

        <div class="section card">
            <h2>Follow-up Queue</h2>
            <div class="section-note">Use this as the callback worklist.</div>
            {dataframe_to_html(data["follow_ups"], "No follow-up calls.")}
        </div>
    </div>
    """

    products_tab = f"""
    <div id="tab-products" class="tab-panel">
        <div class="grid three-col section">
            {bar_chart_html(data["interest_products"].head(10), "product_focus", "interest_calls", "Top Products Generating Interest")}
            {bar_chart_html(data["praise_products"].head(10), "product_focus", "positive_calls", "Top Praised Products")}
            {bar_chart_html(data["follow_up_by_product"].head(10), "product_focus", "count", "Products Needing Follow-up")}
        </div>

        <div class="grid two-col section">
            <div class="card">
                <h2>Products Generating Interest</h2>
                <div class="section-note">Products appearing in enquiry or technical qualification calls.</div>
                {dataframe_to_html(data["interest_products"], "No products with clear interest signals.")}
            </div>
            <div class="card">
                <h2>Products Receiving Positive Praise</h2>
                <div class="section-note">Products customers are happy with and likely to recommend.</div>
                {dataframe_to_html(data["praise_products"], "No praised products found.")}
            </div>
        </div>

        <div class="section card">
            <h2>Product by Call Type Map</h2>
            <div class="section-note">See which products show up under which type of calls.</div>
            {dataframe_to_html(data["type_product_matrix"], "No product/type mapping available.")}
        </div>
    </div>
    """

    issues_tab = f"""
    <div id="tab-issues" class="tab-panel">
        <div class="grid two-col section">
            {bar_chart_html(data["issue_products"].head(10), "product_focus", "issue_calls", "Top Products Generating Issues")}
            {bar_chart_html(data["issue_type_dist"], "type of call", "count", "Issue Load by Call Type")}
        </div>

        <div class="grid two-col section">
            <div class="card">
                <h2>Products Generating Issues</h2>
                <div class="section-note">Support issues, after-sales complaints, unresolved calls, or low sentiment products.</div>
                {dataframe_to_html(data["issue_products"], "No product issues found.")}
            </div>
            <div class="card">
                <h2>Follow-up Queue</h2>
                <div class="section-note">Open items that still require a callback or answer.</div>
                {dataframe_to_html(data["follow_ups"], "No follow-up calls.")}
            </div>
        </div>

        <div class="section card">
            <h2>Bad Calls</h2>
            <div class="section-note">These are the highest-priority calls to review first.</div>
            {dataframe_to_html(data["bad_calls"], "No bad calls found.")}
        </div>
    </div>
    """

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="utf-8"/>
        <meta name="viewport" content="width=device-width, initial-scale=1"/>
        <title>Call Analysis Dashboard</title>
        <style>
            :root {{
                --bg: #0f172a;
                --panel: #111827;
                --panel-2: #1f2937;
                --text: #e5e7eb;
                --muted: #94a3b8;
                --good: #14532d;
                --good-border: #22c55e;
                --bad: #4c0519;
                --bad-border: #f43f5e;
                --warn: #422006;
                --warn-border: #f59e0b;
                --neutral-border: #64748b;
                --accent: #38bdf8;
            }}
            * {{ box-sizing: border-box; }}
            body {{
                margin: 0;
                background: var(--bg);
                color: var(--text);
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
            }}
            .container {{
                max-width: 1500px;
                margin: 0 auto;
                padding: 24px;
            }}
            h1 {{ margin: 0 0 8px; font-size: 32px; }}
            h2 {{ margin: 0 0 12px; font-size: 22px; }}
            h3 {{ margin: 0 0 14px; font-size: 18px; }}
            p.sub {{
                margin: 0 0 20px;
                color: var(--muted);
            }}
            code {{
                background: rgba(255,255,255,0.08);
                padding: 2px 6px;
                border-radius: 6px;
            }}
            .grid {{
                display: grid;
                gap: 16px;
            }}
            .metrics {{
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            }}
            .two-col {{
                grid-template-columns: repeat(auto-fit, minmax(520px, 1fr));
            }}
            .three-col {{
                grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            }}
            .section {{
                margin-top: 18px;
            }}
            .card {{
                background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 18px;
                padding: 18px;
                box-shadow: 0 10px 25px rgba(0, 0, 0, 0.25);
            }}
            .metric {{
                min-height: 112px;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
            }}
            .metric.good {{
                border-color: var(--good-border);
                background: linear-gradient(180deg, rgba(34,197,94,0.18), rgba(17,24,39,0.9));
            }}
            .metric.bad {{
                border-color: var(--bad-border);
                background: linear-gradient(180deg, rgba(244,63,94,0.18), rgba(17,24,39,0.9));
            }}
            .metric.warn {{
                border-color: var(--warn-border);
                background: linear-gradient(180deg, rgba(245,158,11,0.18), rgba(17,24,39,0.9));
            }}
            .metric-title {{
                color: var(--muted);
                font-size: 14px;
            }}
            .metric-value {{
                font-size: 34px;
                font-weight: 700;
                line-height: 1.1;
            }}
            .section-note {{
                color: var(--muted);
                margin-bottom: 12px;
                font-size: 14px;
            }}
            .tabs {{
                display: flex;
                gap: 10px;
                flex-wrap: wrap;
                margin: 20px 0 10px;
            }}
            .tab-button {{
                border: 1px solid rgba(255,255,255,0.12);
                background: rgba(255,255,255,0.04);
                color: var(--text);
                padding: 10px 16px;
                border-radius: 999px;
                cursor: pointer;
                font-size: 14px;
            }}
            .tab-button.active {{
                background: rgba(56,189,248,0.16);
                border-color: rgba(56,189,248,0.45);
            }}
            .tab-panel {{
                display: none;
            }}
            .tab-panel.active {{
                display: block;
            }}
            .table-wrap {{
                overflow-x: auto;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                font-size: 14px;
            }}
            th, td {{
                padding: 12px 10px;
                text-align: left;
                vertical-align: top;
                border-bottom: 1px solid rgba(255,255,255,0.08);
            }}
            th {{
                position: sticky;
                top: 0;
                background: var(--panel-2);
                z-index: 1;
            }}
            tr:hover td {{
                background: rgba(255,255,255,0.02);
            }}
            .empty {{
                color: var(--muted);
                padding: 8px 0;
            }}
            .bars {{
                display: flex;
                flex-direction: column;
                gap: 10px;
            }}
            .bar-row {{
                display: grid;
                grid-template-columns: 180px 1fr 50px;
                gap: 10px;
                align-items: center;
            }}
            .bar-label {{
                font-size: 13px;
                color: var(--text);
                word-break: break-word;
            }}
            .bar-track {{
                background: rgba(255,255,255,0.08);
                border-radius: 999px;
                height: 14px;
                overflow: hidden;
            }}
            .bar-fill {{
                height: 100%;
                background: linear-gradient(90deg, #38bdf8, #818cf8);
                border-radius: 999px;
            }}
            .bar-value {{
                text-align: right;
                color: var(--muted);
                font-size: 13px;
            }}
            .footer {{
                color: var(--muted);
                font-size: 13px;
                margin-top: 24px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Call Analysis Dashboard</h1>
            <p class="sub">
                Reads <code>./output/analysis_results.csv</code> and groups charts into
                <strong>Calls</strong>, <strong>Products</strong>, and <strong>Issues</strong>.
            </p>

            <div class="tabs">
                <button class="tab-button active" onclick="showTab('calls', this)">Calls</button>
                <button class="tab-button" onclick="showTab('products', this)">Products</button>
                <button class="tab-button" onclick="showTab('issues', this)">Issues</button>
            </div>

            {calls_tab}
            {products_tab}
            {issues_tab}

            <div class="footer">
                Refresh the page after regenerating <code>analysis_results.csv</code> to see the latest data.
            </div>
        </div>

        <script>
            function showTab(name, button) {{
                document.querySelectorAll('.tab-panel').forEach(el => el.classList.remove('active'));
                document.querySelectorAll('.tab-button').forEach(el => el.classList.remove('active'));
                document.getElementById('tab-' + name).classList.add('active');
                button.classList.add('active');
            }}
        </script>
    </body>
    </html>
    """


@app.get("/api/summary", response_class=JSONResponse)
def api_summary() -> dict[str, Any]:
    data = build_dashboard_data(CSV_PATH)
    return {
        "metrics": data["metrics"],
        "bad_calls": data["bad_calls"].to_dict(orient="records"),
        "good_calls": data["good_calls"].to_dict(orient="records"),
        "follow_ups": data["follow_ups"].to_dict(orient="records"),
        "interest_products": data["interest_products"].to_dict(orient="records"),
        "issue_products": data["issue_products"].to_dict(orient="records"),
        "praise_products": data["praise_products"].to_dict(orient="records"),
        "type_product_matrix": data["type_product_matrix"].to_dict(orient="records"),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)