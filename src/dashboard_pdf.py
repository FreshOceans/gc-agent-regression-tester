"""Dashboard PDF export utilities."""

from __future__ import annotations

import io
from typing import Optional

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


def _export_with_reportlab(report: TestReport, dashboard_metrics: dict) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter, pageCompression=0)
    width, height = letter
    margin = 48
    y = height - margin

    def write_line(text: str, size: int = 10, bold: bool = False, gap: int = 14) -> None:
        nonlocal y
        font_name = "Helvetica-Bold" if bold else "Helvetica"
        pdf.setFont(font_name, size)
        pdf.drawString(margin, y, text)
        y -= gap

    def ensure_space(min_y: int = 80) -> None:
        nonlocal y
        if y >= min_y:
            return
        pdf.showPage()
        y = height - margin

    kpis = dashboard_metrics.get("kpis", {})
    duration = dashboard_metrics.get("duration", {})
    compare = dashboard_metrics.get("compare")
    outcome_mix = dashboard_metrics.get("outcome_mix", [])
    trend = dashboard_metrics.get("trend", [])
    scenario_health = dashboard_metrics.get("scenario_health", [])
    top_regressions = dashboard_metrics.get("top_regressions", [])

    write_line("GC Agent Regression Tester Dashboard Report", size=16, bold=True, gap=20)
    write_line(f"Suite: {report.suite_name}", size=11, bold=True)
    write_line(f"Generated (UTC): {report.timestamp.isoformat()}")
    write_line(f"Run Duration: {report.duration_seconds:.1f}s")
    write_line("")

    write_line("Executive KPI Summary", size=12, bold=True, gap=16)
    write_line(
        (
            f"Attempts={kpis.get('attempts', 0)}  "
            f"Successes={kpis.get('successes', 0)}  "
            f"Failures={kpis.get('failures', 0)}  "
            f"Timeouts={kpis.get('timeouts', 0)}  "
            f"Skipped={kpis.get('skipped', 0)}"
        )
    )
    write_line(f"Success Rate: {100.0 * float(kpis.get('success_rate', 0.0)):.1f}%")
    write_line(
        (
            f"Avg Duration: {float(duration.get('average_seconds', 0.0)):.2f}s  "
            f"Median: {float(duration.get('median_seconds', 0.0)):.2f}s  "
            f"P95: {float(duration.get('p95_seconds', 0.0)):.2f}s"
        )
    )
    write_line("")

    write_line("Outcome Distribution", size=12, bold=True, gap=16)
    bar_x = margin
    bar_y = y - 8
    bar_width = width - (margin * 2)
    bar_height = 12
    pdf.setStrokeColor(colors.HexColor("#d7dee8"))
    pdf.setFillColor(colors.HexColor("#f1f5f9"))
    pdf.rect(bar_x, bar_y, bar_width, bar_height, fill=1, stroke=1)
    segment_colors = {
        "Success": colors.HexColor("#2e7d32"),
        "Failure": colors.HexColor("#c62828"),
        "Timeout": colors.HexColor("#e65100"),
        "Skipped": colors.HexColor("#4b5563"),
    }
    cursor = bar_x
    for segment in outcome_mix:
        pct = max(0.0, min(1.0, float(segment.get("percentage", 0.0))))
        seg_width = bar_width * pct
        if seg_width <= 0:
            continue
        label = str(segment.get("label", ""))
        pdf.setFillColor(segment_colors.get(label, colors.HexColor("#4a90d9")))
        pdf.rect(cursor, bar_y, seg_width, bar_height, fill=1, stroke=0)
        cursor += seg_width
    y -= 24
    for segment in outcome_mix:
        label = str(segment.get("label", ""))
        pct = 100.0 * float(segment.get("percentage", 0.0))
        count = int(segment.get("count", 0))
        write_line(f"{label}: {count} ({pct:.1f}%)", size=9, gap=12)
    write_line("")

    ensure_space()
    write_line("Current vs Previous Same-Suite Run", size=12, bold=True, gap=16)
    if not compare:
        write_line("No previous same-suite baseline found.")
    else:
        write_line(f"Baseline timestamp (UTC): {compare.get('baseline_timestamp', 'N/A')}")
        deltas = compare.get("deltas", {})
        metric_map = [
            ("success_rate", "Success Rate", True),
            ("failure_rate", "Failure Rate", True),
            ("timeout_rate", "Timeout Rate", True),
            ("skipped_rate", "Skipped Rate", True),
            ("avg_duration_seconds", "Avg Duration", False),
            ("median_duration_seconds", "Median Duration", False),
            ("p95_duration_seconds", "P95 Duration", False),
        ]
        for key, label, is_percent in metric_map:
            delta_row = deltas.get(key)
            if not isinstance(delta_row, dict):
                continue
            cur = float(delta_row.get("current", 0.0))
            base = float(delta_row.get("baseline", 0.0))
            delta = float(delta_row.get("delta", 0.0))
            if is_percent:
                write_line(
                    f"{label}: {100.0 * cur:.1f}% vs {100.0 * base:.1f}% (Δ {100.0 * delta:+.1f}pp)"
                )
            else:
                write_line(f"{label}: {cur:.2f}s vs {base:.2f}s (Δ {delta:+.2f}s)")
    write_line("")

    ensure_space()
    write_line("Recent Same-Suite Trend", size=12, bold=True, gap=16)
    if not trend:
        write_line("No trend history available.")
    else:
        chart_x = margin
        chart_y = y - 80
        chart_w = width - (margin * 2)
        chart_h = 60
        pdf.setStrokeColor(colors.HexColor("#cbd5e1"))
        pdf.rect(chart_x, chart_y, chart_w, chart_h, fill=0, stroke=1)
        points = []
        if len(trend) == 1:
            rate = float(trend[0].get("success_rate", 0.0))
            points = [(chart_x + chart_w / 2.0, chart_y + (chart_h * rate))]
        else:
            for idx, entry in enumerate(trend):
                rate = max(0.0, min(1.0, float(entry.get("success_rate", 0.0))))
                x = chart_x + (chart_w * idx / (len(trend) - 1))
                y_point = chart_y + (chart_h * rate)
                points.append((x, y_point))
        pdf.setStrokeColor(colors.HexColor("#2a6fb8"))
        if len(points) >= 2:
            for idx in range(len(points) - 1):
                pdf.line(points[idx][0], points[idx][1], points[idx + 1][0], points[idx + 1][1])
        pdf.setFillColor(colors.HexColor("#2a6fb8"))
        for x, y_point in points:
            pdf.circle(x, y_point, 2.2, stroke=1, fill=1)
        y = chart_y - 10
        for entry in trend[-5:]:
            write_line(
                (
                    f"{entry.get('timestamp', 'N/A')} · "
                    f"success {100.0 * float(entry.get('success_rate', 0.0)):.1f}%"
                ),
                size=8,
                gap=11,
            )
    write_line("")

    ensure_space()
    write_line("Scenario Deep Dive", size=12, bold=True, gap=16)
    write_line("Name | Success% | Attempts | Failures | Timeouts | Skipped | Regression", size=9, bold=True)
    for row in scenario_health:
        ensure_space(min_y=70)
        write_line(
            (
                f"{row.get('name', '')[:34]} | "
                f"{100.0 * float(row.get('success_rate', 0.0)):.0f}% | "
                f"{int(row.get('attempts', 0))} | "
                f"{int(row.get('failures', 0))} | "
                f"{int(row.get('timeouts', 0))} | "
                f"{int(row.get('skipped', 0))} | "
                f"{'yes' if row.get('is_regression') else 'no'}"
            ),
            size=8,
            gap=11,
        )

    ensure_space()
    write_line("")
    write_line("Top Failing/Timeout Scenarios", size=12, bold=True, gap=16)
    if not top_regressions:
        write_line("No regression-risk scenarios in this run.")
    else:
        for row in top_regressions:
            write_line(
                (
                    f"{row.get('name', '')}: "
                    f"failures={int(row.get('failures', 0))}, "
                    f"timeouts={int(row.get('timeouts', 0))}, "
                    f"skipped={int(row.get('skipped', 0))}, "
                    f"success={100.0 * float(row.get('success_rate', 0.0)):.1f}%"
                )
            )

    pdf.save()
    return buffer.getvalue()


def _export_with_fallback_pdf(report: TestReport, dashboard_metrics: dict) -> bytes:
    """Create a minimal valid PDF without third-party dependencies."""
    kpis = dashboard_metrics.get("kpis", {})
    duration = dashboard_metrics.get("duration", {})
    compare = dashboard_metrics.get("compare")
    lines = [
        "GC Agent Regression Tester Dashboard Report",
        f"Suite: {report.suite_name}",
        f"Generated (UTC): {report.timestamp.isoformat()}",
        f"Run Duration: {report.duration_seconds:.1f}s",
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
            f"Avg Duration: {float(duration.get('average_seconds', 0.0)):.2f}s "
            f"Median: {float(duration.get('median_seconds', 0.0)):.2f}s "
            f"P95: {float(duration.get('p95_seconds', 0.0)):.2f}s"
        ),
        "",
        "Current vs Previous Same-Suite Run",
    ]
    if not compare:
        lines.append("No previous same-suite baseline found.")
    else:
        lines.append(f"Baseline timestamp (UTC): {compare.get('baseline_timestamp', 'N/A')}")
        for key, label in [
            ("success_rate", "Success Rate"),
            ("failure_rate", "Failure Rate"),
            ("timeout_rate", "Timeout Rate"),
            ("skipped_rate", "Skipped Rate"),
            ("avg_duration_seconds", "Avg Duration"),
            ("median_duration_seconds", "Median Duration"),
            ("p95_duration_seconds", "P95 Duration"),
        ]:
            delta = compare.get("deltas", {}).get(key, {})
            lines.append(
                f"{label}: current={delta.get('current', 0)} baseline={delta.get('baseline', 0)} delta={delta.get('delta', 0)}"
            )
    lines.extend(
        [
            "",
            "Scenario Deep Dive",
            "Name | Success% | Attempts | Failures | Timeouts | Skipped | Regression",
        ]
    )
    for row in dashboard_metrics.get("scenario_health", []):
        lines.append(
            (
                f"{row.get('name', '')[:48]} | "
                f"{100.0 * float(row.get('success_rate', 0.0)):.0f}% | "
                f"{int(row.get('attempts', 0))} | "
                f"{int(row.get('failures', 0))} | "
                f"{int(row.get('timeouts', 0))} | "
                f"{int(row.get('skipped', 0))} | "
                f"{'yes' if row.get('is_regression') else 'no'}"
            )
        )
    return _simple_text_pdf(lines)


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
