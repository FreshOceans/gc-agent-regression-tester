"""Dashboard PDF export utilities."""

from __future__ import annotations

import io

from .duration_format import format_duration, format_duration_delta
from .models import TestReport


def export_dashboard_pdf(
    report: TestReport,
    dashboard_metrics: dict,
) -> bytes:
    """Export dashboard summary as PDF bytes.

    Uses reportlab when available for richer visuals. Falls back to a small
    pure-Python PDF writer to keep server-side export available.
    """
    try:
        return _export_with_reportlab(report, dashboard_metrics)
    except Exception:
        return _export_with_fallback_pdf(report, dashboard_metrics)


def _styles():
    return {
        "colors": {
            "bg": "#f8fafc",
            "panel": "#ffffff",
            "border": "#d7dee8",
            "text": "#1f2937",
            "muted": "#64748b",
            "title": "#0f172a",
            "brand": "#1a1a2e",
            "brand_light": "#4a90d9",
            "success": "#2e7d32",
            "failure": "#c62828",
            "timeout": "#e65100",
            "skipped": "#4b5563",
            "rate": "#0b5394",
            "card_bg": "#f9fbfd",
            "track": "#e6ebf2",
            "regression": "#b91c1c",
        },
        "font": {
            "title": 16,
            "h2": 12,
            "label": 9,
            "body": 9,
            "metric": 15,
            "small": 8,
        },
        "space": {
            "section": 14,
            "card_gap": 8,
            "line": 12,
        },
    }


def _export_with_reportlab(report: TestReport, dashboard_metrics: dict) -> bytes:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter, pageCompression=0)
    width, height = letter
    margin = 42
    y = height - margin

    styles = _styles()

    kpis = dashboard_metrics.get("kpis", {})
    duration = dashboard_metrics.get("duration", {})
    compare = dashboard_metrics.get("compare")
    outcome_mix = dashboard_metrics.get("outcome_mix", [])
    trend = dashboard_metrics.get("trend", [])
    flakiness = dashboard_metrics.get("flakiness", {})
    scenario_health = dashboard_metrics.get("scenario_health", [])
    scenario_tool_health = dashboard_metrics.get("scenario_tool_health", [])
    tool_effectiveness = dashboard_metrics.get("tool_effectiveness", {})
    journey_effectiveness = dashboard_metrics.get("journey_effectiveness", {})
    top_regressions = dashboard_metrics.get("top_regressions", [])

    # Page 1: Executive dashboard
    y = draw_title_header(
        pdf,
        report,
        margin=margin,
        page_width=width,
        y_top=y,
        styles=styles,
    )
    y = draw_kpi_cards(
        pdf,
        kpis,
        duration,
        margin=margin,
        page_width=width,
        y_top=y,
        styles=styles,
    )
    y = draw_outcome_mix(
        pdf,
        outcome_mix,
        margin=margin,
        page_width=width,
        y_top=y,
        styles=styles,
    )
    y = draw_compare_panel(
        pdf,
        compare,
        margin=margin,
        page_width=width,
        y_top=y,
        styles=styles,
    )
    y = draw_tool_effectiveness_panel(
        pdf,
        tool_effectiveness,
        journey_effectiveness,
        compare,
        margin=margin,
        page_width=width,
        y_top=y,
        styles=styles,
    )
    draw_trend_panel(
        pdf,
        trend,
        margin=margin,
        page_width=width,
        y_top=y,
        styles=styles,
    )

    # Page 2: Scenario analytics
    pdf.showPage()
    y = height - margin
    y = draw_title_header(
        pdf,
        report,
        margin=margin,
        page_width=width,
        y_top=y,
        styles=styles,
        subtitle="Scenario Analytics",
    )
    y = draw_scenario_league_table(
        pdf,
        scenario_health,
        scenario_tool_health=scenario_tool_health,
        margin=margin,
        page_width=width,
        y_top=y,
        styles=styles,
    )
    y = draw_top_regressions(
        pdf,
        top_regressions,
        margin=margin,
        page_width=width,
        y_top=y,
        styles=styles,
    )
    y = draw_flakiness_panel(
        pdf,
        flakiness,
        margin=margin,
        page_width=width,
        y_top=y,
        styles=styles,
    )

    if not compare:
        _draw_footer_note(
            pdf,
            "Baseline note: No previous same-suite baseline found.",
            margin=margin,
            page_width=width,
            y=margin + 10,
            styles=styles,
        )

    pdf.save()
    return buffer.getvalue()


def draw_title_header(
    pdf,
    report: TestReport,
    *,
    margin: float,
    page_width: float,
    y_top: float,
    styles: dict,
    subtitle: str = "Executive Dashboard",
) -> float:
    from reportlab.lib import colors

    colors_map = styles["colors"]
    fonts = styles["font"]

    header_h = 72
    x = margin
    y = y_top - header_h
    w = page_width - (margin * 2)

    pdf.setFillColor(colors.HexColor(colors_map["brand"]))
    pdf.setStrokeColor(colors.HexColor(colors_map["brand"]))
    pdf.roundRect(x, y, w, header_h, 10, fill=1, stroke=1)

    pdf.setFillColor(colors.white)
    pdf.setFont("Helvetica-Bold", fonts["title"])
    pdf.drawString(x + 14, y + header_h - 22, "Regression Test Harness Dashboard Report")

    pdf.setFont("Helvetica", fonts["label"])
    pdf.drawString(x + 14, y + header_h - 36, f"Suite: {report.suite_name}")
    pdf.drawString(x + 14, y + header_h - 49, f"Generated (UTC): {report.timestamp.isoformat()}")
    pdf.drawString(x + 14, y + header_h - 62, f"Run Duration: {format_duration(report.duration_seconds, 1)}")

    pdf.setFont("Helvetica-Bold", fonts["label"])
    pdf.drawRightString(x + w - 14, y + header_h - 36, subtitle)

    return y - styles["space"]["section"]


def draw_kpi_cards(
    pdf,
    kpis: dict,
    duration: dict,
    *,
    margin: float,
    page_width: float,
    y_top: float,
    styles: dict,
) -> float:
    from reportlab.lib import colors

    colors_map = styles["colors"]
    fonts = styles["font"]
    gap = styles["space"]["card_gap"]

    _draw_section_title(pdf, "Executive KPI Summary", margin, y_top, styles)
    y = y_top - 14

    labels = [
        ("Total Attempts", int(kpis.get("attempts", 0)), colors_map["brand"]),
        ("Successes", int(kpis.get("successes", 0)), colors_map["success"]),
        ("Failures", int(kpis.get("failures", 0)), colors_map["failure"]),
        ("Timeouts", int(kpis.get("timeouts", 0)), colors_map["timeout"]),
        ("Skipped", int(kpis.get("skipped", 0)), colors_map["skipped"]),
        ("Success Rate", f"{100.0 * float(kpis.get('success_rate', 0.0)):.1f}%", colors_map["rate"]),
    ]

    cols = 3
    card_w = (page_width - (margin * 2) - (gap * (cols - 1))) / cols
    card_h = 56

    for idx, (label, value, accent) in enumerate(labels):
        row = idx // cols
        col = idx % cols
        x = margin + col * (card_w + gap)
        card_y = y - (row + 1) * card_h - row * gap
        _draw_metric_card(
            pdf,
            x,
            card_y,
            card_w,
            card_h,
            label,
            str(value),
            accent,
            styles,
        )

    y = y - (2 * card_h) - gap - 10

    duration_cards = [
        ("Average", format_duration(float(duration.get("average_seconds", 0.0)), 2)),
        ("Median", format_duration(float(duration.get("median_seconds", 0.0)), 2)),
        ("P95", format_duration(float(duration.get("p95_seconds", 0.0)), 2)),
    ]

    mini_h = 38
    for idx, (label, value) in enumerate(duration_cards):
        x = margin + idx * (card_w + gap)
        card_y = y - mini_h
        pdf.setFillColor(colors.HexColor(colors_map["card_bg"]))
        pdf.setStrokeColor(colors.HexColor(colors_map["border"]))
        pdf.roundRect(x, card_y, card_w, mini_h, 6, fill=1, stroke=1)
        pdf.setFillColor(colors.HexColor(colors_map["muted"]))
        pdf.setFont("Helvetica", fonts["small"])
        pdf.drawString(x + 8, card_y + mini_h - 13, label)
        pdf.setFillColor(colors.HexColor(colors_map["text"]))
        pdf.setFont("Helvetica-Bold", fonts["label"])
        pdf.drawString(x + 8, card_y + 10, value)

    return y - mini_h - styles["space"]["section"]


def draw_outcome_mix(
    pdf,
    outcome_mix: list[dict],
    *,
    margin: float,
    page_width: float,
    y_top: float,
    styles: dict,
) -> float:
    from reportlab.lib import colors

    colors_map = styles["colors"]
    fonts = styles["font"]

    _draw_section_title(pdf, "Outcome Mix", margin, y_top, styles)
    y = y_top - 16

    panel_h = 78
    panel_y = y - panel_h
    panel_w = page_width - (margin * 2)
    _draw_panel(pdf, margin, panel_y, panel_w, panel_h, styles)

    bar_x = margin + 12
    bar_y = panel_y + panel_h - 27
    bar_w = panel_w - 24
    bar_h = 12

    pdf.setFillColor(colors.HexColor(colors_map["track"]))
    pdf.setStrokeColor(colors.HexColor(colors_map["border"]))
    pdf.roundRect(bar_x, bar_y, bar_w, bar_h, 6, fill=1, stroke=1)

    segment_colors = {
        "Success": colors_map["success"],
        "Failure": colors_map["failure"],
        "Timeout": colors_map["timeout"],
        "Skipped": colors_map["skipped"],
    }

    cursor = bar_x
    for segment in outcome_mix:
        pct = _clamp(float(segment.get("percentage", 0.0)), 0.0, 1.0)
        seg_w = bar_w * pct
        if seg_w <= 0:
            continue
        color = segment_colors.get(str(segment.get("label", "")), colors_map["brand_light"])
        pdf.setFillColor(colors.HexColor(color))
        pdf.rect(cursor, bar_y, seg_w, bar_h, fill=1, stroke=0)
        cursor += seg_w

    chip_x = bar_x
    chip_y = panel_y + 14
    for segment in outcome_mix:
        label = str(segment.get("label", ""))
        count = int(segment.get("count", 0))
        pct = 100.0 * float(segment.get("percentage", 0.0))
        text = f"{label}: {count} ({pct:.1f}%)"
        text_w = pdf.stringWidth(text, "Helvetica", fonts["small"])
        chip_w = text_w + 18
        chip_h = 14

        color = segment_colors.get(label, colors_map["brand_light"])
        pdf.setFillColor(colors.HexColor("#ffffff"))
        pdf.setStrokeColor(colors.HexColor(colors_map["border"]))
        pdf.roundRect(chip_x, chip_y, chip_w, chip_h, 6, fill=1, stroke=1)

        pdf.setFillColor(colors.HexColor(color))
        pdf.circle(chip_x + 6, chip_y + (chip_h / 2), 2.2, fill=1, stroke=0)

        pdf.setFillColor(colors.HexColor(colors_map["text"]))
        pdf.setFont("Helvetica", fonts["small"])
        pdf.drawString(chip_x + 11, chip_y + 4, text)

        chip_x += chip_w + 6
        if chip_x > (margin + panel_w - 120):
            chip_x = bar_x
            chip_y -= chip_h + 4

    return panel_y - styles["space"]["section"]


def draw_compare_panel(
    pdf,
    compare: dict | None,
    *,
    margin: float,
    page_width: float,
    y_top: float,
    styles: dict,
) -> float:
    from reportlab.lib import colors

    colors_map = styles["colors"]
    fonts = styles["font"]

    _draw_section_title(pdf, "Current vs Previous Same-Suite Run", margin, y_top, styles)
    y = y_top - 16

    panel_h = 106
    panel_y = y - panel_h
    panel_w = page_width - (margin * 2)
    _draw_panel(pdf, margin, panel_y, panel_w, panel_h, styles)

    if not compare:
        pdf.setFont("Helvetica", fonts["body"])
        pdf.setFillColor(colors.HexColor(colors_map["muted"]))
        pdf.drawString(margin + 12, panel_y + panel_h / 2, "No previous same-suite baseline found.")
        return panel_y - styles["space"]["section"]

    pdf.setFont("Helvetica", fonts["small"])
    pdf.setFillColor(colors.HexColor(colors_map["muted"]))
    pdf.drawString(
        margin + 12,
        panel_y + panel_h - 15,
        f"Baseline suite: {compare.get('baseline_suite_name', 'N/A')}",
    )
    pdf.drawString(
        margin + 12,
        panel_y + panel_h - 25,
        f"Storage: {compare.get('baseline_storage_type', 'full_json')}",
    )
    pdf.drawString(
        margin + 12,
        panel_y + panel_h - 35,
        f"Baseline timestamp (UTC): {compare.get('baseline_timestamp', 'N/A')}",
    )

    metric_map = [
        ("Success Rate", "success_rate", True),
        ("Failure Rate", "failure_rate", True),
        ("Timeout Rate", "timeout_rate", True),
        ("Skipped Rate", "skipped_rate", True),
        ("Avg Duration", "avg_duration_seconds", False),
        ("Median", "median_duration_seconds", False),
        ("P95", "p95_duration_seconds", False),
    ]
    deltas = compare.get("deltas", {}) if isinstance(compare, dict) else {}

    start_x = margin + 12
    row_y = panel_y + panel_h - 48
    line_h = 11
    col_gap = (panel_w - 24) / 2

    for idx, (label, key, is_percent) in enumerate(metric_map):
        row = idx % 4
        col = idx // 4
        x = start_x + col * col_gap
        y_line = row_y - row * line_h

        delta_row = deltas.get(key, {}) if isinstance(deltas.get(key, {}), dict) else {}
        current_value = float(delta_row.get("current", 0.0) or 0.0)
        baseline_value = float(delta_row.get("baseline", 0.0) or 0.0)
        delta_value = float(delta_row.get("delta", 0.0) or 0.0)

        if is_percent:
            right_text = f"{100.0 * current_value:.1f}% vs {100.0 * baseline_value:.1f}% (Δ {100.0 * delta_value:+.1f}pp)"
        else:
            right_text = (
                f"{format_duration(current_value, 2)} vs {format_duration(baseline_value, 2)} "
                f"(Δ {format_duration_delta(delta_value, 2)})"
            )

        direction = "flat"
        if delta_value > 0:
            direction = "up"
        elif delta_value < 0:
            direction = "down"
        color = {
            "up": colors_map["success"],
            "down": colors_map["failure"],
            "flat": colors_map["muted"],
        }[direction]

        pdf.setFont("Helvetica-Bold", fonts["small"])
        pdf.setFillColor(colors.HexColor(colors_map["text"]))
        pdf.drawString(x, y_line, f"{label}:")

        pdf.setFont("Helvetica", fonts["small"])
        pdf.setFillColor(colors.HexColor(color))
        pdf.drawString(x + 50, y_line, right_text)

    return panel_y - styles["space"]["section"]


def draw_flakiness_panel(
    pdf,
    flakiness: dict | None,
    *,
    margin: float,
    page_width: float,
    y_top: float,
    styles: dict,
) -> float:
    from reportlab.lib import colors

    colors_map = styles["colors"]
    fonts = styles["font"]

    _draw_section_title(pdf, "Unstable Scenarios", margin, y_top, styles)
    y = y_top - 16

    panel_h = 74
    panel_y = y - panel_h
    panel_w = page_width - (margin * 2)
    _draw_panel(pdf, margin, panel_y, panel_w, panel_h, styles)

    unstable = []
    if isinstance(flakiness, dict):
        unstable = flakiness.get("unstable_scenarios", []) or []

    if not unstable:
        pdf.setFont("Helvetica", fonts["body"])
        pdf.setFillColor(colors.HexColor(colors_map["muted"]))
        pdf.drawString(
            margin + 12,
            panel_y + panel_h / 2,
            "Not enough same-suite history to compute stability risk.",
        )
        return panel_y - styles["space"]["section"]

    runs = int(flakiness.get("evaluated_runs", 0)) if isinstance(flakiness, dict) else 0
    scenarios = int(flakiness.get("scenarios_evaluated", 0)) if isinstance(flakiness, dict) else 0
    pdf.setFont("Helvetica", fonts["small"])
    pdf.setFillColor(colors.HexColor(colors_map["muted"]))
    pdf.drawString(
        margin + 12,
        panel_y + panel_h - 14,
        f"Evaluated {scenarios} scenarios across {runs} run(s)",
    )

    row_y = panel_y + panel_h - 27
    for index, row in enumerate(unstable[:3]):
        label = f"{index + 1}. {_truncate(str(row.get('name', '')), 28)}"
        reason = str(row.get("reason", ""))
        score = float(row.get("instability_score", 0.0) or 0.0)
        pdf.setFillColor(colors.HexColor(colors_map["text"]))
        pdf.setFont("Helvetica-Bold", fonts["small"])
        pdf.drawString(margin + 12, row_y, label)
        pdf.setFillColor(colors.HexColor(colors_map["muted"]))
        pdf.setFont("Helvetica", fonts["small"])
        pdf.drawString(
            margin + 180,
            row_y,
            f"{_truncate(reason, 38)} (score {score:.2f})",
        )
        row_y -= 13

    return panel_y - styles["space"]["section"]


def draw_tool_effectiveness_panel(
    pdf,
    tool_effectiveness: dict | None,
    journey_effectiveness: dict | None,
    compare: dict | None,
    *,
    margin: float,
    page_width: float,
    y_top: float,
    styles: dict,
) -> float:
    from reportlab.lib import colors

    colors_map = styles["colors"]
    fonts = styles["font"]

    _draw_section_title(pdf, "Tool Effectiveness", margin, y_top, styles)
    y = y_top - 16

    panel_h = 102
    panel_y = y - panel_h
    panel_w = page_width - (margin * 2)
    _draw_panel(pdf, margin, panel_y, panel_w, panel_h, styles)

    metrics = tool_effectiveness or {}
    validated_attempts = int(metrics.get("validated_attempts", 0) or 0)
    journey_metrics = journey_effectiveness or {}
    journey_validated = int(journey_metrics.get("validated_attempts", 0) or 0)

    if validated_attempts <= 0 and journey_validated <= 0:
        pdf.setFont("Helvetica", fonts["body"])
        pdf.setFillColor(colors.HexColor(colors_map["muted"]))
        pdf.drawString(
            margin + 12,
            panel_y + panel_h / 2,
            "No tool or journey validation scenarios were configured in this run.",
        )
        return panel_y - styles["space"]["section"]

    loose_rate = float(metrics.get("loose_pass_rate", 0.0) or 0.0)
    strict_rate = float(metrics.get("strict_pass_rate", 0.0) or 0.0)
    missing_signal = int(metrics.get("missing_signal_count", 0) or 0)
    order_mismatch = int(metrics.get("order_mismatch_count", 0) or 0)

    row_top = panel_y + panel_h - 16
    pdf.setFont("Helvetica", fonts["small"])
    pdf.setFillColor(colors.HexColor(colors_map["text"]))
    pdf.drawString(margin + 12, row_top, f"Validated Attempts: {validated_attempts}")
    pdf.drawString(margin + 170, row_top, f"Loose Pass Rate: {100.0 * loose_rate:.1f}%")
    pdf.drawString(margin + 338, row_top, f"Strict Pass Rate: {100.0 * strict_rate:.1f}%")

    pdf.drawString(margin + 12, row_top - 13, f"Missing Signal: {missing_signal}")
    pdf.drawString(margin + 170, row_top - 13, f"Order Mismatch: {order_mismatch}")

    journey_passes = int(journey_metrics.get("passes", 0) or 0)
    journey_contained = int(journey_metrics.get("contained_passes", 0) or 0)
    journey_fulfilled = int(journey_metrics.get("fulfillment_passes", 0) or 0)
    journey_path = int(journey_metrics.get("path_passes", 0) or 0)
    journey_category = int(journey_metrics.get("category_match_passes", 0) or 0)

    pdf.setFillColor(colors.HexColor(colors_map["text"]))
    pdf.drawString(
        margin + 12,
        row_top - 30,
        f"Journey Validated: {journey_validated}",
    )
    pdf.drawString(
        margin + 170,
        row_top - 30,
        f"Journey Passes: {journey_passes}",
    )
    pdf.drawString(
        margin + 338,
        row_top - 30,
        f"Contained Passes: {journey_contained}",
    )
    pdf.drawString(
        margin + 12,
        row_top - 43,
        f"Fulfillment Passes: {journey_fulfilled}",
    )
    pdf.drawString(
        margin + 170,
        row_top - 43,
        f"Path Passes: {journey_path}",
    )
    pdf.drawString(
        margin + 338,
        row_top - 43,
        f"Category Match Passes: {journey_category}",
    )

    compare_deltas = compare.get("deltas", {}) if isinstance(compare, dict) else {}
    loose_delta = compare_deltas.get("tool_loose_pass_rate", {})
    strict_delta = compare_deltas.get("tool_strict_pass_rate", {})

    if isinstance(loose_delta, dict):
        direction = str(loose_delta.get("direction", "flat"))
        delta_value = float(loose_delta.get("delta", 0.0) or 0.0)
        color = colors_map["muted"]
        if direction == "up":
            color = colors_map["success"]
        elif direction == "down":
            color = colors_map["failure"]
        pdf.setFillColor(colors.HexColor(color))
        pdf.drawString(
            margin + 12,
            row_top - 60,
            f"Loose Δ vs baseline: {100.0 * delta_value:+.1f}pp",
        )

    if isinstance(strict_delta, dict):
        direction = str(strict_delta.get("direction", "flat"))
        delta_value = float(strict_delta.get("delta", 0.0) or 0.0)
        color = colors_map["muted"]
        if direction == "up":
            color = colors_map["success"]
        elif direction == "down":
            color = colors_map["failure"]
        pdf.setFillColor(colors.HexColor(color))
        pdf.drawString(
            margin + 220,
            row_top - 60,
            f"Strict Δ vs baseline: {100.0 * delta_value:+.1f}pp",
        )

    return panel_y - styles["space"]["section"]


def draw_trend_panel(
    pdf,
    trend: list[dict],
    *,
    margin: float,
    page_width: float,
    y_top: float,
    styles: dict,
) -> float:
    from reportlab.lib import colors

    colors_map = styles["colors"]
    fonts = styles["font"]

    _draw_section_title(pdf, "Recent Same-Suite Trend", margin, y_top, styles)
    y = y_top - 16

    panel_h = 86
    panel_y = y - panel_h
    panel_w = page_width - (margin * 2)
    _draw_panel(pdf, margin, panel_y, panel_w, panel_h, styles)

    if not trend:
        pdf.setFont("Helvetica", fonts["body"])
        pdf.setFillColor(colors.HexColor(colors_map["muted"]))
        pdf.drawString(margin + 12, panel_y + panel_h / 2, "No trend history available.")
        return panel_y - styles["space"]["section"]

    chart_x = margin + 12
    chart_y = panel_y + 18
    chart_w = panel_w - 24
    chart_h = panel_h - 34

    pdf.setStrokeColor(colors.HexColor(colors_map["border"]))
    pdf.rect(chart_x, chart_y, chart_w, chart_h, fill=0, stroke=1)

    points = []
    if len(trend) == 1:
        rate = _clamp(float(trend[0].get("success_rate", 0.0)), 0.0, 1.0)
        points.append((chart_x + chart_w / 2.0, chart_y + chart_h * rate, bool(trend[0].get("is_current"))))
    else:
        for idx, item in enumerate(trend):
            rate = _clamp(float(item.get("success_rate", 0.0)), 0.0, 1.0)
            px = chart_x + (chart_w * idx / (len(trend) - 1))
            py = chart_y + chart_h * rate
            points.append((px, py, bool(item.get("is_current"))))

    pdf.setStrokeColor(colors.HexColor(colors_map["brand_light"]))
    if len(points) >= 2:
        for idx in range(len(points) - 1):
            pdf.line(points[idx][0], points[idx][1], points[idx + 1][0], points[idx + 1][1])

    for px, py, is_current in points:
        dot_color = colors_map["success"] if is_current else colors_map["brand_light"]
        pdf.setFillColor(colors.HexColor(dot_color))
        pdf.circle(px, py, 2.5, fill=1, stroke=0)

    latest = trend[-1]
    latest_label = f"Latest success: {100.0 * float(latest.get('success_rate', 0.0)):.1f}%"
    pdf.setFont("Helvetica", fonts["small"])
    pdf.setFillColor(colors.HexColor(colors_map["muted"]))
    pdf.drawRightString(margin + panel_w - 12, panel_y + 7, latest_label)

    return panel_y - styles["space"]["section"]


def draw_scenario_league_table(
    pdf,
    scenario_health: list[dict],
    scenario_tool_health: list[dict] | None = None,
    *,
    margin: float,
    page_width: float,
    y_top: float,
    styles: dict,
) -> float:
    from reportlab.lib import colors

    colors_map = styles["colors"]
    fonts = styles["font"]

    _draw_section_title(pdf, "Scenario League Table", margin, y_top, styles)
    y = y_top - 16

    panel_w = page_width - (margin * 2)
    row_h = 19
    header_h = 18
    max_table_h = 330

    total_rows_possible = max(1, int((max_table_h - header_h - 8) // row_h))
    rows = scenario_health[:total_rows_possible]
    remaining = max(0, len(scenario_health) - len(rows))
    panel_h = header_h + (len(rows) * row_h) + (18 if remaining else 8)

    panel_y = y - panel_h
    _draw_panel(pdf, margin, panel_y, panel_w, panel_h, styles)

    header_y = panel_y + panel_h - header_h + 4
    pdf.setFont("Helvetica-Bold", fonts["small"])
    pdf.setFillColor(colors.HexColor(colors_map["muted"]))
    pdf.drawString(margin + 10, header_y, "Scenario")
    pdf.drawString(margin + 248, header_y, "Success")
    pdf.drawString(margin + 338, header_y, "Attempts")
    pdf.drawString(margin + 392, header_y, "F/T/S")
    pdf.drawString(margin + 452, header_y, "Reg")
    pdf.drawString(margin + 490, header_y, "Tool")

    tool_lookup = {
        str(row.get("name", "")).strip().lower(): row
        for row in (scenario_tool_health or [])
        if str(row.get("name", "")).strip()
    }

    y_cursor = panel_y + panel_h - header_h - 4
    for row in rows:
        y_cursor -= row_h
        name = str(row.get("name", ""))
        success_rate = _clamp(float(row.get("success_rate", 0.0)), 0.0, 1.0)
        attempts = int(row.get("attempts", 0))
        failures = int(row.get("failures", 0))
        timeouts = int(row.get("timeouts", 0))
        skipped = int(row.get("skipped", 0))
        is_regression = bool(row.get("is_regression"))

        pdf.setStrokeColor(colors.HexColor("#edf1f5"))
        pdf.line(margin + 8, y_cursor - 1, margin + panel_w - 8, y_cursor - 1)

        pdf.setFont("Helvetica", fonts["small"])
        pdf.setFillColor(colors.HexColor(colors_map["text"]))
        pdf.drawString(margin + 10, y_cursor + 5, _truncate(name, 34))

        bar_x = margin + 248
        bar_y = y_cursor + 3
        bar_w = 80
        bar_h = 8
        pdf.setFillColor(colors.HexColor(colors_map["track"]))
        pdf.roundRect(bar_x, bar_y, bar_w, bar_h, 4, fill=1, stroke=0)
        fill_color = colors_map["regression"] if is_regression else colors_map["success"]
        pdf.setFillColor(colors.HexColor(fill_color))
        pdf.roundRect(bar_x, bar_y, bar_w * success_rate, bar_h, 4, fill=1, stroke=0)
        pdf.setFillColor(colors.HexColor(colors_map["text"]))
        pdf.drawRightString(bar_x + bar_w + 26, y_cursor + 5, f"{success_rate * 100:.0f}%")

        pdf.drawString(margin + 338, y_cursor + 5, str(attempts))
        pdf.drawString(margin + 392, y_cursor + 5, f"{failures}/{timeouts}/{skipped}")

        reg_color = colors_map["regression"] if is_regression else colors_map["success"]
        reg_text = "yes" if is_regression else "no"
        pdf.setFillColor(colors.HexColor(reg_color))
        pdf.drawString(margin + 452, y_cursor + 5, reg_text)

        tool_row = tool_lookup.get(name.strip().lower())
        if tool_row is None:
            pdf.setFillColor(colors.HexColor(colors_map["muted"]))
            pdf.drawString(margin + 490, y_cursor + 5, "n/a")
        else:
            loose_rate = 100.0 * float(tool_row.get("tool_loose_pass_rate", 0.0) or 0.0)
            strict_rate = 100.0 * float(tool_row.get("tool_strict_pass_rate", 0.0) or 0.0)
            validated_attempts = int(tool_row.get("tool_validated_attempts", 0) or 0)
            pdf.setFillColor(colors.HexColor(colors_map["text"]))
            pdf.drawString(
                margin + 490,
                y_cursor + 5,
                f"L{loose_rate:.0f}/S{strict_rate:.0f} ({validated_attempts})",
            )

    if remaining:
        pdf.setFont("Helvetica", fonts["small"])
        pdf.setFillColor(colors.HexColor(colors_map["muted"]))
        pdf.drawString(margin + 10, panel_y + 8, f"... plus {remaining} more scenarios")

    return panel_y - styles["space"]["section"]


def draw_top_regressions(
    pdf,
    top_regressions: list[dict],
    *,
    margin: float,
    page_width: float,
    y_top: float,
    styles: dict,
) -> float:
    from reportlab.lib import colors

    colors_map = styles["colors"]
    fonts = styles["font"]

    _draw_section_title(pdf, "Top Failing/Timeout Scenarios", margin, y_top, styles)
    y = y_top - 16

    panel_h = 130
    panel_y = y - panel_h
    panel_w = page_width - (margin * 2)
    _draw_panel(pdf, margin, panel_y, panel_w, panel_h, styles)

    if not top_regressions:
        pdf.setFont("Helvetica", fonts["body"])
        pdf.setFillColor(colors.HexColor(colors_map["muted"]))
        pdf.drawString(margin + 12, panel_y + panel_h / 2, "No regression-risk scenarios in this run.")
        return panel_y - styles["space"]["section"]

    y_cursor = panel_y + panel_h - 16
    max_rows = min(5, len(top_regressions))
    for row in top_regressions[:max_rows]:
        name = _truncate(str(row.get("name", "")), 46)
        failures = int(row.get("failures", 0))
        timeouts = int(row.get("timeouts", 0))
        skipped = int(row.get("skipped", 0))
        success_rate = 100.0 * float(row.get("success_rate", 0.0))

        pdf.setFont("Helvetica-Bold", fonts["small"])
        pdf.setFillColor(colors.HexColor(colors_map["text"]))
        pdf.drawString(margin + 12, y_cursor, name)

        pdf.setFont("Helvetica", fonts["small"])
        pdf.setFillColor(colors.HexColor(colors_map["muted"]))
        pdf.drawString(
            margin + 12,
            y_cursor - 10,
            f"failures={failures}, timeouts={timeouts}, skipped={skipped}, success={success_rate:.1f}%",
        )

        y_cursor -= 22

    return panel_y - styles["space"]["section"]


def _export_with_fallback_pdf(report: TestReport, dashboard_metrics: dict) -> bytes:
    """Create a minimal valid PDF without third-party dependencies."""
    kpis = dashboard_metrics.get("kpis", {})
    duration = dashboard_metrics.get("duration", {})
    compare = dashboard_metrics.get("compare")
    trend = dashboard_metrics.get("trend", [])
    lines = [
        "Regression Test Harness Dashboard Report",
        "Executive Dashboard",
        f"Suite: {report.suite_name}",
        f"Generated (UTC): {report.timestamp.isoformat()}",
        f"Run Duration: {format_duration(report.duration_seconds, 1)}",
        "",
        "Executive KPI Summary",
        (
            f"Attempts={kpis.get('attempts', 0)} "
            f"Successes={kpis.get('successes', 0)} "
            f"Failures={kpis.get('failures', 0)} "
            f"Timeouts={kpis.get('timeouts', 0)} "
            f"Skipped={kpis.get('skipped', 0)}"
        ),
        f"Success Rate: {100.0 * float(kpis.get('success_rate', 0.0)):.1f}%",
        (
            f"Avg Duration: {format_duration(float(duration.get('average_seconds', 0.0)), 2)} "
            f"Median: {format_duration(float(duration.get('median_seconds', 0.0)), 2)} "
            f"P95: {format_duration(float(duration.get('p95_seconds', 0.0)), 2)}"
        ),
        "",
        "Outcome Mix",
    ]

    outcome_mix = dashboard_metrics.get("outcome_mix", [])
    for segment in outcome_mix:
        lines.append(
            f"{segment.get('label', '')}: {int(segment.get('count', 0))} ({100.0 * float(segment.get('percentage', 0.0)):.1f}%)"
        )

    lines.extend(["", "Current vs Previous Same-Suite Run"])
    if not compare:
        lines.append("No previous same-suite baseline found.")
    else:
        lines.append(f"Baseline suite: {compare.get('baseline_suite_name', 'N/A')}")
        lines.append(f"Baseline storage: {compare.get('baseline_storage_type', 'full_json')}")
        lines.append(f"Baseline timestamp (UTC): {compare.get('baseline_timestamp', 'N/A')}")
        for key, label in [
            ("success_rate", "Success Rate"),
            ("failure_rate", "Failure Rate"),
            ("timeout_rate", "Timeout Rate"),
            ("skipped_rate", "Skipped Rate"),
            ("avg_duration_seconds", "Avg Duration"),
            ("median_duration_seconds", "Median Duration"),
            ("p95_duration_seconds", "P95 Duration"),
            ("tool_loose_pass_rate", "Tool Loose Pass Rate"),
            ("tool_strict_pass_rate", "Tool Strict Pass Rate"),
        ]:
            delta = compare.get("deltas", {}).get(key, {})
            current_value = float(delta.get("current", 0) or 0)
            baseline_value = float(delta.get("baseline", 0) or 0)
            delta_value = float(delta.get("delta", 0) or 0)
            if key.endswith("_rate") or key == "success_rate":
                lines.append(
                    (
                        f"{label}: current={100.0 * current_value:.1f}% "
                        f"baseline={100.0 * baseline_value:.1f}% "
                        f"delta={100.0 * delta_value:+.1f}pp"
                    )
                )
            else:
                lines.append(
                    (
                        f"{label}: current={format_duration(current_value, 2)} "
                        f"baseline={format_duration(baseline_value, 2)} "
                        f"delta={format_duration_delta(delta_value, 2)}"
                    )
                )

    lines.append("")
    lines.append("Tool Effectiveness")
    tool_effectiveness = dashboard_metrics.get("tool_effectiveness", {})
    validated_attempts = int(tool_effectiveness.get("validated_attempts", 0) or 0)
    if validated_attempts <= 0:
        lines.append("No tool-validation scenarios were configured in this run.")
    else:
        lines.append(
            (
                f"Validated={validated_attempts}, "
                f"Loose={100.0 * float(tool_effectiveness.get('loose_pass_rate', 0.0)):.1f}%, "
                f"Strict={100.0 * float(tool_effectiveness.get('strict_pass_rate', 0.0)):.1f}%, "
                f"MissingSignal={int(tool_effectiveness.get('missing_signal_count', 0) or 0)}, "
                f"OrderMismatch={int(tool_effectiveness.get('order_mismatch_count', 0) or 0)}"
            )
        )

    lines.append("")
    lines.append("Journey Effectiveness")
    journey_effectiveness = dashboard_metrics.get("journey_effectiveness", {})
    journey_validated = int(journey_effectiveness.get("validated_attempts", 0) or 0)
    if journey_validated <= 0:
        lines.append("No journey-validation scenarios were configured in this run.")
    else:
        lines.append(
            (
                f"Validated={journey_validated}, "
                f"Passes={int(journey_effectiveness.get('passes', 0) or 0)}, "
                f"Contained={int(journey_effectiveness.get('contained_passes', 0) or 0)}, "
                f"Fulfillment={int(journey_effectiveness.get('fulfillment_passes', 0) or 0)}, "
                f"Path={int(journey_effectiveness.get('path_passes', 0) or 0)}, "
                f"CategoryMatch={int(journey_effectiveness.get('category_match_passes', 0) or 0)}"
            )
        )

    lines.append("")
    lines.append("Unstable Scenarios")
    flakiness = dashboard_metrics.get("flakiness", {})
    unstable_rows = flakiness.get("unstable_scenarios", []) if isinstance(flakiness, dict) else []
    if not unstable_rows:
        lines.append("Not enough same-suite history to compute stability risk.")
    else:
        lines.append(
            f"Evaluated {int(flakiness.get('scenarios_evaluated', 0))} scenarios across {int(flakiness.get('evaluated_runs', 0))} run(s)."
        )
        for row in unstable_rows[:5]:
            lines.append(
                f"{row.get('name', '')}: {row.get('reason', '')} (score {float(row.get('instability_score', 0.0)):.2f})"
            )

    lines.append("")
    lines.append("Recent Same-Suite Trend")
    if not trend:
        lines.append("No trend history available.")
    else:
        for entry in trend[-5:]:
            lines.append(
                f"{entry.get('timestamp', 'N/A')} · success {100.0 * float(entry.get('success_rate', 0.0)):.1f}%"
            )

    lines.extend([
        "",
        "Scenario League Table",
        "Name | Success% | Attempts | Failures | Timeouts | Skipped | Regression | Tool(L/S)",
    ])
    for row in dashboard_metrics.get("scenario_health", []):
        tool_note = "n/a"
        validated = int(row.get("tool_validated_attempts", 0) or 0)
        if validated > 0:
            tool_note = (
                f"{100.0 * float(row.get('tool_loose_pass_rate', 0.0)):.0f}%/"
                f"{100.0 * float(row.get('tool_strict_pass_rate', 0.0)):.0f}%"
            )
        lines.append(
            (
                f"{row.get('name', '')[:48]} | "
                f"{100.0 * float(row.get('success_rate', 0.0)):.0f}% | "
                f"{int(row.get('attempts', 0))} | "
                f"{int(row.get('failures', 0))} | "
                f"{int(row.get('timeouts', 0))} | "
                f"{int(row.get('skipped', 0))} | "
                f"{'yes' if row.get('is_regression') else 'no'} | "
                f"{tool_note}"
            )
        )

    lines.append("")
    lines.append("Top Failing/Timeout Scenarios")
    top_regressions = dashboard_metrics.get("top_regressions", [])
    if not top_regressions:
        lines.append("No regression-risk scenarios in this run.")
    else:
        for row in top_regressions:
            lines.append(
                (
                    f"{row.get('name', '')}: "
                    f"failures={int(row.get('failures', 0))}, "
                    f"timeouts={int(row.get('timeouts', 0))}, "
                    f"skipped={int(row.get('skipped', 0))}, "
                    f"success={100.0 * float(row.get('success_rate', 0.0)):.1f}%"
                )
            )

    if not compare:
        lines.extend(["", "Baseline note: No previous same-suite baseline found."])

    return _simple_text_pdf(lines)


def _draw_section_title(pdf, title: str, x: float, y_top: float, styles: dict) -> None:
    from reportlab.lib import colors

    pdf.setFont("Helvetica-Bold", styles["font"]["h2"])
    pdf.setFillColor(colors.HexColor(styles["colors"]["title"]))
    pdf.drawString(x, y_top, title)


def _draw_panel(pdf, x: float, y: float, w: float, h: float, styles: dict) -> None:
    from reportlab.lib import colors

    pdf.setFillColor(colors.HexColor(styles["colors"]["panel"]))
    pdf.setStrokeColor(colors.HexColor(styles["colors"]["border"]))
    pdf.roundRect(x, y, w, h, 8, fill=1, stroke=1)


def _draw_metric_card(
    pdf,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    value: str,
    accent_color: str,
    styles: dict,
) -> None:
    from reportlab.lib import colors

    colors_map = styles["colors"]
    fonts = styles["font"]

    pdf.setFillColor(colors.HexColor(colors_map["card_bg"]))
    pdf.setStrokeColor(colors.HexColor(colors_map["border"]))
    pdf.roundRect(x, y, w, h, 6, fill=1, stroke=1)

    pdf.setFillColor(colors.HexColor(accent_color))
    pdf.rect(x + 1, y + h - 4, w - 2, 3, fill=1, stroke=0)

    pdf.setFillColor(colors.HexColor(colors_map["muted"]))
    pdf.setFont("Helvetica", fonts["small"])
    pdf.drawString(x + 8, y + h - 15, label)

    pdf.setFillColor(colors.HexColor(colors_map["text"]))
    pdf.setFont("Helvetica-Bold", fonts["metric"])
    pdf.drawString(x + 8, y + 12, value)


def _draw_footer_note(
    pdf,
    text: str,
    *,
    margin: float,
    page_width: float,
    y: float,
    styles: dict,
) -> None:
    from reportlab.lib import colors

    panel_w = page_width - (margin * 2)
    panel_h = 20
    _draw_panel(pdf, margin, y, panel_w, panel_h, styles)

    pdf.setFillColor(colors.HexColor(styles["colors"]["muted"]))
    pdf.setFont("Helvetica", styles["font"]["small"])
    pdf.drawString(margin + 8, y + 6, text)


def _truncate(value: str, max_length: int) -> str:
    if len(value) <= max_length:
        return value
    return value[: max(0, max_length - 1)] + "…"


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _simple_text_pdf(lines: list[str]) -> bytes:
    """Build a small single-page PDF with text lines."""
    y_start = 780
    line_gap = 14
    content = ["BT", "/F1 10 Tf", f"50 {y_start} Td"]
    for index, line in enumerate(lines):
        if index > 0:
            content.append(f"0 -{line_gap} Td")
        safe = (
            str(line)
            .replace("\\", "\\\\")
            .replace("(", "\\(")
            .replace(")", "\\)")
        )
        content.append(f"({safe}) Tj")
    content.append("ET")
    stream_bytes = "\n".join(content).encode("latin-1", "replace")

    objects = []

    def obj(number: int, payload: bytes) -> bytes:
        return f"{number} 0 obj\n".encode() + payload + b"\nendobj\n"

    objects.append(obj(1, b"<< /Type /Catalog /Pages 2 0 R >>"))
    objects.append(obj(2, b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>"))
    objects.append(
        obj(
            3,
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        )
    )
    objects.append(
        obj(
            4,
            f"<< /Length {len(stream_bytes)} >>\nstream\n".encode()
            + stream_bytes
            + b"\nendstream",
        )
    )
    objects.append(obj(5, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"))

    header = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
    body = b""
    offsets = [0]
    current = len(header)
    for raw in objects:
        offsets.append(current)
        body += raw
        current += len(raw)

    xref_pos = len(header) + len(body)
    xref = [f"xref\n0 {len(objects) + 1}\n".encode(), b"0000000000 65535 f \n"]
    for offset in offsets[1:]:
        xref.append(f"{offset:010d} 00000 n \n".encode())
    xref_bytes = b"".join(xref)

    trailer = (
        b"trailer\n"
        + f"<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode()
        + b"startxref\n"
        + f"{xref_pos}\n".encode()
        + b"%%EOF\n"
    )
    return header + body + xref_bytes + trailer
