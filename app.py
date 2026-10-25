# -*- coding: utf-8 -*-
"""
DTC Dual-Endpoint Prediction Tool
Shiny for Python UI v5 — three-column manuscript layout

Backend:
- Locked prediction_engine.py
- Efficacy: 5 MI-specific CatBoost models
- Injury: 5 MI-specific Logistic Regression models
- Patient-level mean probability across 5 MI
- Locked Training mean-OOF Youden thresholds

This app performs prediction only.
It does not retrain, tune, recalibrate, or reselect thresholds.
"""

from __future__ import annotations

import math
from typing import Any

from shiny import App, reactive, render, ui

from prediction_engine import (
    EFFICACY_THRESHOLD,
    INJURY_THRESHOLD,
    model_integrity_qc,
    predict_patient,
)


# ---------------------------------------------------------------------
# Visual constants
# ---------------------------------------------------------------------
COLORS = {
    "deep_blue": "#5271AE",
    "light_blue": "#70ACDE",
    "yellow": "#F5CC7D",
    "orange": "#FFA660",
    "red": "#D85B59",
    "gray": "#B8B8B8",
    "dark": "#253247",
    "muted": "#667085",
    "panel": "#FFFFFF",
    "page": "#F5F7FB",
    "border": "#E4E7EC",
}

JOINT_COLORS = {
    "ER + No injury": COLORS["deep_blue"],
    "ER + Injury": COLORS["light_blue"],
    "Non-ER + No injury": COLORS["orange"],
    "Non-ER + Injury": COLORS["red"],
}


def pct(x: float) -> str:
    return f"{100.0 * float(x):.1f}%"


def threshold_pct(x: float) -> str:
    return f"{100.0 * float(x):.1f}%"


def probability_ring(
    probability: float,
    label: str,
    subtitle: str,
    color: str,
) -> ui.Tag:
    p = max(0.0, min(1.0, float(probability)))
    deg = 360.0 * p

    return ui.div(
        ui.div(
            ui.div(
                ui.div(
                    ui.div(pct(p), class_="ring-value"),
                    ui.div(label, class_="ring-label"),
                    class_="ring-inner",
                ),
                class_="probability-ring",
                style=(
                    f"background: conic-gradient("
                    f"{color} 0deg {deg:.3f}deg, "
                    f"#EDF1F7 {deg:.3f}deg 360deg);"
                ),
            ),
            class_="ring-wrap",
        ),
        ui.div(subtitle, class_="ring-subtitle"),
        class_="risk-ring-card",
    )


def result_badge(text: str, kind: str) -> ui.Tag:
    if kind == "good":
        color = COLORS["deep_blue"]
        bg = "#EEF3FF"
    elif kind == "warning":
        color = COLORS["orange"]
        bg = "#FFF5E8"
    else:
        color = COLORS["red"]
        bg = "#FFF0EF"

    return ui.span(
        text,
        class_="result-badge",
        style=f"color:{color}; background:{bg};",
    )


def joint_map(prob_non_er: float, prob_injury: float) -> ui.Tag:
    """
    Clean 2D probability map.
    X-axis: Non-ER probability
    Y-axis: Injury probability
    """
    x = max(0.0, min(1.0, float(prob_non_er)))
    y = max(0.0, min(1.0, float(prob_injury)))

    x_pct = 100.0 * x
    y_top = 100.0 * (1.0 - y)

    x_thr = 100.0 * EFFICACY_THRESHOLD
    y_thr_top = 100.0 * (1.0 - INJURY_THRESHOLD)

    return ui.div(
        ui.div(
            # Four subtle quadrants
            ui.div(
                class_="quad-bg",
                style=(
                    f"left:0;top:0;width:{x_thr:.4f}%;height:{y_thr_top:.4f}%;"
                    f"background:#D7E9F6;"
                ),
            ),
            ui.div(
                class_="quad-bg",
                style=(
                    f"left:{x_thr:.4f}%;top:0;width:{100-x_thr:.4f}%;"
                    f"height:{y_thr_top:.4f}%;background:#F3C0BE;"
                ),
            ),
            ui.div(
                class_="quad-bg",
                style=(
                    f"left:0;top:{y_thr_top:.4f}%;width:{x_thr:.4f}%;"
                    f"height:{100-y_thr_top:.4f}%;background:#C8D6EE;"
                ),
            ),
            ui.div(
                class_="quad-bg",
                style=(
                    f"left:{x_thr:.4f}%;top:{y_thr_top:.4f}%;"
                    f"width:{100-x_thr:.4f}%;height:{100-y_thr_top:.4f}%;"
                    f"background:#F8D2AA;"
                ),
            ),

            # Threshold lines
            ui.div(
                class_="threshold-line-v",
                style=f"left:{x_thr:.4f}%;",
            ),
            ui.div(
                class_="threshold-line-h",
                style=f"top:{y_thr_top:.4f}%;",
            ),

            # Labels
            ui.div("ER + Injury", class_="quad-label quad-tl"),
            ui.div("Non-ER + Injury", class_="quad-label quad-tr"),
            ui.div("ER + No injury", class_="quad-label quad-bl"),
            ui.div("Non-ER + No injury", class_="quad-label quad-br"),

            # Patient marker
            ui.div(
                class_="patient-point",
                style=(
                    f"left:calc({x_pct:.4f}% - 8px);"
                    f"top:calc({y_top:.4f}% - 8px);"
                ),
                title=(
                    f"Non-ER probability: {pct(x)}; "
                    f"Injury probability: {pct(y)}"
                ),
            ),
            class_="joint-map-area",
        ),
        ui.div(
            ui.span("0%", class_="axis-left"),
            ui.span("Non-ER probability", class_="axis-title-x"),
            ui.span("100%", class_="axis-right"),
            class_="axis-x-row",
        ),
        ui.div("Injury probability", class_="axis-title-y"),
        ui.div(
            ui.span(
                ui.span(class_="legend-dot", style="background:#C8D6EE;"),
                " ER + No injury",
            ),
            ui.span(
                ui.span(class_="legend-dot", style="background:#D7E9F6;"),
                " ER + Injury",
            ),
            ui.span(
                ui.span(class_="legend-dot", style="background:#F8D2AA;"),
                " Non-ER + No injury",
            ),
            ui.span(
                ui.span(class_="legend-dot", style="background:#F3C0BE;"),
                " Non-ER + Injury",
            ),
            class_="joint-legend",
        ),
        class_="joint-map-shell",
    )

# ---------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------
APP_CSS = f"""
:root {{
  --deep-blue: {COLORS["deep_blue"]};
  --light-blue: {COLORS["light_blue"]};
  --orange: {COLORS["orange"]};
  --red: {COLORS["red"]};
  --dark: {COLORS["dark"]};
  --muted: {COLORS["muted"]};
  --border: {COLORS["border"]};
  --page: {COLORS["page"]};
}}

body {{
  background: var(--page);
  color: var(--dark);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
}}

.app-shell {{
  max-width: 1740px;
  margin: 0 auto;
  padding: 22px 22px 36px 22px;
}}

.topbar {{
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 20px;
  margin-bottom: 16px;
}}

.app-title {{
  font-size: 28px;
  line-height: 1.2;
  font-weight: 760;
  margin: 0 0 6px 0;
  color: #1E293B;
}}

.app-subtitle {{
  font-size: 14px;
  line-height: 1.5;
  color: var(--muted);
  max-width: 900px;
}}

.model-lock {{
  font-size: 12px;
  color: var(--deep-blue);
  background: #EEF3FF;
  border: 1px solid #D7E2FF;
  border-radius: 999px;
  padding: 7px 11px;
  white-space: nowrap;
}}

/* ------------------------------------------------------------
   Main desktop layout: 26% clinical/tumor | 26% salivary | 48% results
   ------------------------------------------------------------ */
.main-grid {{
  display: grid;
  grid-template-columns:
    minmax(280px, 26fr)
    minmax(280px, 26fr)
    minmax(560px, 48fr);
  gap: 16px;
  align-items: stretch;
}}

.panel {{
  background: #FFFFFF;
  border: 1px solid var(--border);
  border-radius: 18px;
  box-shadow: 0 8px 28px rgba(16,24,40,.05);
  min-width: 0;
}}

.input-panel {{
  padding: 18px 18px 16px 18px;
}}

.output-panel {{
  padding: 18px 18px 16px 18px;
  min-width: 0;
}}

/* Result panel typography: slightly larger for manuscript/web readability */
.output-panel .section-title {{
  font-size: 19px;
  font-weight: 760;
}}

.output-panel .section-note {{
  font-size: 12.5px;
  line-height: 1.45;
}}

.column-title,
.section-title {{
  font-size: 18px;
  font-weight: 740;
  line-height: 1.25;
  margin: 0 0 5px 0;
  color: #253247;
}}

.column-note,
.section-note {{
  font-size: 11.5px;
  line-height: 1.45;
  color: var(--muted);
  margin-bottom: 13px;
}}

.column-divider {{
  height: 1px;
  background: #F0F2F5;
  margin: 12px 0 14px 0;
}}

.form-group {{
  margin-bottom: 9px;
}}

.form-label {{
  font-size: 12.5px;
  line-height: 1.3;
  font-weight: 600;
  color: #344054;
  margin-bottom: 5px;
}}

.form-control,
.form-select {{
  border-radius: 10px !important;
  border-color: #D0D5DD !important;
  min-height: 36px;
  padding-top: 6px !important;
  padding-bottom: 6px !important;
  box-shadow: none !important;
  font-size: 13px;
}}

.form-control:focus,
.form-select:focus {{
  border-color: var(--deep-blue) !important;
  box-shadow: 0 0 0 3px rgba(82,113,174,.12) !important;
}}

.action-block {{
  border-top: 1px solid #F0F2F5;
  margin-top: 13px;
  padding-top: 13px;
}}

.action-row {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}}

.btn-primary {{
  width: 100%;
  min-height: 42px;
  border-radius: 11px;
  border: none !important;
  background: var(--deep-blue) !important;
  font-weight: 700;
  margin-top: 0;
}}

.btn-outline-secondary {{
  width: 100%;
  min-height: 42px;
  border-radius: 11px;
  margin-top: 0;
}}

.results-placeholder {{
  min-height: 620px;
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
  color: #98A2B3;
  padding: 30px;
}}

.placeholder-icon {{
  width: 58px;
  height: 58px;
  border-radius: 50%;
  margin: 0 auto 14px auto;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #EEF3FF;
  color: var(--deep-blue);
  font-size: 25px;
  font-weight: 700;
}}

.risk-grid {{
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}}

.risk-card {{
  min-width: 0;
  border: 1px solid var(--border);
  border-radius: 15px;
  padding: 14px 14px 12px 14px;
  background: #FFFFFF;
}}

.risk-card-title {{
  font-size: 15.5px;
  font-weight: 740;
  color: #344054;
  margin-bottom: 2px;
}}

.risk-card-model {{
  font-size: 11.5px;
  color: #98A2B3;
  margin-bottom: 7px;
}}

.risk-card-row {{
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;
}}

.risk-ring-card {{
  width: 100%;
}}

.ring-wrap {{
  display: flex;
  justify-content: center;
}}

.probability-ring {{
  width: 104px;
  height: 104px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: inset 0 0 0 1px rgba(15,23,42,.03);
}}

.ring-inner {{
  width: 76px;
  height: 76px;
  background: #FFFFFF;
  border-radius: 50%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}}

.ring-value {{
  font-size: 21px;
  font-weight: 800;
  color: #1D2939;
  line-height: 1.05;
}}

.ring-label {{
  font-size: 9.5px;
  line-height: 1.15;
  color: #667085;
  margin-top: 2px;
  text-align: center;
}}

.ring-subtitle {{
  font-size: 10.8px;
  color: #98A2B3;
  text-align: center;
  margin-top: 5px;
}}

.risk-metadata {{
  width: 100%;
  display: grid;
  grid-template-columns: 1fr;
  gap: 3px;
  text-align: center;
}}

.meta-line {{
  font-size: 11.8px;
  line-height: 1.35;
  color: #667085;
  margin: 0;
}}

.meta-line strong {{
  color: #344054;
}}

.result-badge {{
  display: inline-block;
  padding: 5px 9px;
  border-radius: 999px;
  font-size: 11.5px;
  font-weight: 760;
  margin-left: 4px;
}}

.joint-card {{
  margin-top: 12px;
  border: 1px solid var(--border);
  border-radius: 15px;
  padding: 14px;
}}

.joint-summary {{
  display: flex;
  gap: 12px;
  align-items: flex-start;
  justify-content: space-between;
  margin-bottom: 10px;
}}

.joint-profile-title {{
  font-size: 13.5px;
  color: #667085;
}}

.joint-profile-value {{
  font-size: 19px;
  font-weight: 800;
  margin-top: 2px;
}}

.joint-map-shell {{
  position: relative;
  padding-left: 32px;
  padding-bottom: 22px;
}}

.joint-map-area {{
  height: 225px;
  position: relative;
  border: 1px solid #D0D5DD;
  border-radius: 11px;
  overflow: hidden;
  background: #FFFFFF;
}}

.quad-bg {{
  position: absolute;
  z-index: 0;
}}

.threshold-line-v {{
  position: absolute;
  z-index: 3;
  top: 0;
  bottom: 0;
  width: 0;
  border-left: 1.8px dashed #5F6B7A;
}}

.threshold-line-h {{
  position: absolute;
  z-index: 3;
  left: 0;
  right: 0;
  height: 0;
  border-top: 1.8px dashed #5F6B7A;
}}

.patient-point {{
  position: absolute;
  z-index: 5;
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: #111827;
  border: 3px solid #FFFFFF;
  box-shadow: 0 0 0 3px rgba(17,24,39,.24);
}}

.quad-label {{
  position: absolute;
  z-index: 2;
  font-size: 10.8px;
  font-weight: 700;
  color: rgba(31,41,55,.82);
  pointer-events: none;
}}

.quad-tl {{ left: 9px; top: 7px; }}
.quad-tr {{ right: 9px; top: 7px; }}
.quad-bl {{ left: 9px; bottom: 7px; }}
.quad-br {{ right: 9px; bottom: 7px; }}

.axis-x-row {{
  position: relative;
  font-size: 10.5px;
  color: #98A2B3;
  margin-top: 5px;
  height: 16px;
}}

.axis-left {{
  position: absolute;
  left: 0;
}}

.axis-right {{
  position: absolute;
  right: 0;
}}

.axis-title-x {{
  position: absolute;
  left: 50%;
  transform: translateX(-50%);
  color: #667085;
  font-weight: 600;
}}

.axis-title-y {{
  position: absolute;
  left: 0;
  top: 50%;
  transform: rotate(-90deg) translate(-50%, -50%);
  transform-origin: left top;
  color: #667085;
  font-size: 10.5px;
  font-weight: 600;
  white-space: nowrap;
}}

.joint-legend {{
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 6px 12px;
  margin-top: 7px;
  font-size: 10.5px;
  color: #667085;
}}

.legend-dot {{
  display: inline-block;
  width: 9px;
  height: 9px;
  border-radius: 2px;
  margin-right: 4px;
  vertical-align: -1px;
  border: 1px solid rgba(31,41,55,.08);
}}

/* Compact, collapsed model/deployment information */
.model-details {{
  margin-top: 11px;
  background: #F8FAFC;
  border: 1px solid #EAECF0;
  border-radius: 11px;
  color: #667085;
}}

.model-details summary {{
  cursor: pointer;
  padding: 9px 11px;
  font-size: 10.5px;
  font-weight: 650;
  color: #667085;
  user-select: none;
}}

.audit-box {{
  padding: 0 11px 10px 11px;
  font-size: 9.5px;
  color: #667085;
  line-height: 1.5;
}}

.error-box {{
  background: #FFF4F2;
  border: 1px solid #FECDCA;
  color: #B42318;
  border-radius: 12px;
  padding: 14px;
  font-size: 13px;
  line-height: 1.55;
}}

.footer-note {{
  margin-top: 10px;
  font-size: 9.5px;
  line-height: 1.45;
  color: #98A2B3;
}}

/* Two input columns + full-width results on medium screens */
@media (max-width: 1280px) {{
  .app-shell {{
    max-width: 1180px;
  }}

  .main-grid {{
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }}

  .output-panel {{
    grid-column: 1 / -1;
  }}
}}

/* Single-column mobile/tablet layout */
@media (max-width: 760px) {{
  .app-shell {{
    padding: 16px 10px 28px 10px;
  }}

  .topbar {{
    flex-direction: column;
  }}

  .main-grid {{
    grid-template-columns: 1fr;
  }}

  .output-panel {{
    grid-column: auto;
  }}

  .risk-grid {{
    grid-template-columns: 1fr;
  }}

  .joint-summary {{
    flex-direction: column;
  }}

  .joint-map-area {{
    height: 210px;
  }}

  .action-row {{
    grid-template-columns: 1fr;
  }}
}}
"""


# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------
app_ui = ui.page_fluid(
    ui.tags.style(APP_CSS),
    ui.div(
        # ---------------------------------------------------------
        # Header
        # ---------------------------------------------------------
        ui.div(
            ui.div(
                ui.h1(
                    "DTC Dual-Endpoint Prediction Tool",
                    class_="app-title",
                ),
                ui.div(
                    "Prediction of treatment response and salivary gland "
                    "injury after first radioactive iodine therapy in "
                    "differentiated thyroid cancer.",
                    class_="app-subtitle",
                ),
            ),
            class_="topbar",
        ),

        # ---------------------------------------------------------
        # Three-column desktop layout
        # ---------------------------------------------------------
        ui.div(
            # =====================================================
            # COLUMN 1: Treatment & tumor characteristics
            # =====================================================
            ui.div(
                ui.div(
                    "Treatment & Tumor Characteristics",
                    class_="column-title",
                ),
                ui.div(
                    "Required pre-treatment clinical and tumor variables.",
                    class_="column-note",
                ),
                ui.div(class_="column-divider"),

                ui.input_numeric(
                    "first_rai_activity",
                    "First RAI activity (mCi)",
                    value=None,
                    min=0,
                ),
                ui.input_numeric(
                    "number_metastatic",
                    "Number of metastatic lymph nodes",
                    value=None,
                    min=0,
                    step=1,
                ),
                ui.input_numeric(
                    "pre_rai_tg",
                    "Pre-RAI Tg (ng/mL)",
                    value=None,
                    min=0,
                ),
                ui.input_numeric(
                    "surgery_to_rai_days",
                    "Surgery-to-RAI interval (days)",
                    value=None,
                    min=0,
                ),
                ui.input_numeric(
                    "pre_rai_tgab",
                    "Pre-RAI TgAb (IU/mL)",
                    value=None,
                    min=0,
                ),
                ui.input_select(
                    "surgery_type",
                    "Surgery type",
                    choices={
                        "": "Select...",
                        "Total": "Total thyroidectomy",
                        "Near-total": "Near-total thyroidectomy",
                    },
                    selected="",
                ),
                ui.input_select(
                    "ata2025_risk",
                    "2025 ATA recurrence risk",
                    choices={
                        "": "Select...",
                        "Low": "Low",
                        "Low-intermediate": "Low-intermediate",
                        "Intermediate-high": "Intermediate-high",
                        "High": "High",
                    },
                    selected="",
                ),
                ui.input_numeric(
                    "max_ln_metastasis",
                    "Maximum metastatic lymph node size (cm)",
                    value=None,
                    min=0,
                ),
                ui.input_numeric(
                    "bmi",
                    "BMI (kg/m²)",
                    value=None,
                    min=0,
                ),
                class_="panel input-panel",
            ),

            # =====================================================
            # COLUMN 2: Baseline salivary gland function
            # =====================================================
            ui.div(
                ui.div(
                    "Baseline Salivary Gland Function",
                    class_="column-title",
                ),
                ui.div(
                    "Required baseline salivary scintigraphy variables.",
                    class_="column-note",
                ),
                ui.div(class_="column-divider"),

                ui.input_numeric(
                    "lp_peak_count",
                    "Left parotid peak count (counts)",
                    value=None,
                    min=0,
                ),
                ui.input_numeric(
                    "rp_peak_count",
                    "Right parotid peak count (counts)",
                    value=None,
                    min=0,
                ),
                ui.input_select(
                    "parotid_vitc_response_count",
                    "Parotid glands with decreased counts after vitamin C",
                    choices={
                        "": "Select...",
                        "0": "0 — Neither gland",
                        "1": "1 — One gland",
                        "2": "2 — Both glands",
                    },
                    selected="",
                ),
                ui.input_select(
                    "rsm_t_half_status",
                    "Right submandibular gland T1/2 status",
                    choices={
                        "": "Select...",
                        "Measurable": "Measurable",
                        "Prolonged beyond observation window":
                            "Prolonged beyond observation window",
                    },
                    selected="",
                ),
                ui.output_ui("rsm_t_half_input"),
                ui.input_numeric(
                    "rsm_ef",
                    "Right submandibular gland EF (%)",
                    value=None,
                    min=0,
                    max=100,
                ),

                # Calculate after both input groups are completed.
                ui.div(
                    ui.div(
                        ui.input_action_button(
                            "calculate",
                            "Calculate risk",
                            class_="btn-primary",
                        ),
                        ui.input_action_button(
                            "reset",
                            "Reset",
                            class_="btn-outline-secondary",
                        ),
                        class_="action-row",
                    ),
                    class_="action-block",
                ),
                class_="panel input-panel",
            ),

            # =====================================================
            # COLUMN 3: Prediction results
            # =====================================================
            ui.div(
                ui.output_ui("results_panel"),
                class_="panel output-panel",
            ),

            class_="main-grid",
        ),
        class_="app-shell",
    ),
)


# ---------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------
def server(input, output, session):

    # Keep latest successful prediction.
    prediction_state = reactive.Value(None)
    error_state = reactive.Value(None)

    @render.ui
    def rsm_t_half_input():
        if input.rsm_t_half_status() == "Measurable":
            return ui.input_numeric(
                "rsm_t_half",
                "Right submandibular gland T1/2 (min)",
                value=None,
                min=0,
            )
        if (
            input.rsm_t_half_status()
            == "Prolonged beyond observation window"
        ):
            return ui.div(
                "Continuous T1/2 input is not required for a prolonged "
                "observation-window status.",
                class_="section-note",
                style="margin-top:-4px; margin-bottom:8px;",
            )
        return ui.div()

    def current_raw_input() -> dict[str, Any]:
        return {
            "first_rai_activity": input.first_rai_activity(),
            "number_metastatic": input.number_metastatic(),
            "pre_rai_tg": input.pre_rai_tg(),
            "surgery_to_rai_days": input.surgery_to_rai_days(),
            "pre_rai_tgab": input.pre_rai_tgab(),
            "surgery_type": input.surgery_type(),
            "ata2025_risk": input.ata2025_risk(),
            "max_ln_metastasis": input.max_ln_metastasis(),
            "bmi": input.bmi(),
            "lp_peak_count": input.lp_peak_count(),
            "rp_peak_count": input.rp_peak_count(),
            "parotid_vitc_response_count": input.parotid_vitc_response_count(),
            "rsm_t_half_status": input.rsm_t_half_status(),
            "rsm_t_half": (
                input.rsm_t_half()
                if input.rsm_t_half_status() == "Measurable"
                else None
            ),
            "rsm_ef": input.rsm_ef(),
        }

    @reactive.effect
    @reactive.event(input.calculate)
    def _calculate():
        try:
            raw = current_raw_input()
            pred = predict_patient(raw)
            prediction_state.set(pred)
            error_state.set(None)
        except Exception as e:
            prediction_state.set(None)
            error_state.set(str(e))

    @reactive.effect
    @reactive.event(input.reset)
    def _reset():
        prediction_state.set(None)
        error_state.set(None)

        ui.update_numeric("first_rai_activity", value=None)
        ui.update_numeric("number_metastatic", value=None)
        ui.update_numeric("pre_rai_tg", value=None)
        ui.update_numeric("surgery_to_rai_days", value=None)
        ui.update_numeric("pre_rai_tgab", value=None)
        ui.update_select("surgery_type", selected="")
        ui.update_select("ata2025_risk", selected="")
        ui.update_numeric("max_ln_metastasis", value=None)
        ui.update_numeric("bmi", value=None)
        ui.update_numeric("lp_peak_count", value=None)
        ui.update_numeric("rp_peak_count", value=None)
        ui.update_select("parotid_vitc_response_count", selected="")
        ui.update_select("rsm_t_half_status", selected="")
        ui.update_numeric("rsm_ef", value=None)

    @render.ui
    def results_panel():
        err = error_state.get()
        pred = prediction_state.get()

        if err:
            return ui.div(
                ui.div(
                    "Input error",
                    class_="section-title",
                    style="margin-bottom:10px;",
                ),
                ui.div(err, class_="error-box"),
            )

        if pred is None:
            qc = model_integrity_qc()
            qc_text = (
                "Backend model integrity QC: passed"
                if qc["passed"]
                else "Backend model integrity QC: failed"
            )

            return ui.div(
                ui.div(
                    ui.div("↗", class_="placeholder-icon"),
                    ui.div(
                        "Prediction results will appear here",
                        class_="section-title",
                    ),
                    ui.div(
                        "Complete the required variables and click "
                        "“Calculate risk”.",
                        class_="section-note",
                        style="margin-top:6px;",
                    ),
                    ui.div(
                        qc_text,
                        class_="model-lock",
                        style="display:inline-block; margin-top:8px;",
                    ),
                ),
                class_="results-placeholder",
            )

        pe = float(pred["efficacy"]["mean_probability"])
        pi = float(pred["injury"]["mean_probability"])

        eff_class = pred["efficacy"]["predicted_class"]
        inj_class = pred["injury"]["predicted_class"]
        joint = pred["joint_profile"]

        eff_badge = result_badge(
            eff_class,
            "good" if eff_class == "ER" else "risk",
        )
        inj_badge = result_badge(
            inj_class,
            "good" if inj_class == "No injury" else "risk",
        )

        joint_color = JOINT_COLORS.get(
            joint, COLORS["deep_blue"]
        )

        return ui.div(
            ui.div(
                "Prediction Results",
                class_="section-title",
            ),
            ui.div(
                "Probabilities are the mean of predictions from five "
                "imputation-specific locked models.",
                class_="section-note",
            ),

            ui.div(
                # Efficacy
                ui.div(
                    ui.div(
                        "Treatment response",
                        class_="risk-card-title",
                    ),
                    ui.div(
                        "Locked model: CatBoost",
                        class_="risk-card-model",
                    ),
                    ui.div(
                        probability_ring(
                            pe,
                            "Non-ER risk",
                            "Predicted probability",
                            COLORS["orange"],
                        ),
                        ui.div(
                            ui.div(
                                ui.strong("Classification: "),
                                eff_badge,
                                class_="meta-line",
                            ),
                            ui.div(
                                ui.strong("Decision threshold: "),
                                threshold_pct(EFFICACY_THRESHOLD),
                                class_="meta-line",
                            ),
                            ui.div(
                                ui.strong("ER probability: "),
                                pct(1.0 - pe),
                                class_="meta-line",
                            ),
                            class_="risk-metadata",
                        ),
                        class_="risk-card-row",
                    ),
                    class_="risk-card",
                ),

                # Injury
                ui.div(
                    ui.div(
                        "Salivary gland injury",
                        class_="risk-card-title",
                    ),
                    ui.div(
                        "Locked model: Logistic Regression",
                        class_="risk-card-model",
                    ),
                    ui.div(
                        probability_ring(
                            pi,
                            "Injury risk",
                            "Predicted probability",
                            COLORS["red"],
                        ),
                        ui.div(
                            ui.div(
                                ui.strong("Classification: "),
                                inj_badge,
                                class_="meta-line",
                            ),
                            ui.div(
                                ui.strong("Decision threshold: "),
                                threshold_pct(INJURY_THRESHOLD),
                                class_="meta-line",
                            ),
                            ui.div(
                                ui.strong("No-injury probability: "),
                                pct(1.0 - pi),
                                class_="meta-line",
                            ),
                            class_="risk-metadata",
                        ),
                        class_="risk-card-row",
                    ),
                    class_="risk-card",
                ),
                class_="risk-grid",
            ),

            # Joint profile
            ui.div(
                ui.div(
                    ui.div(
                        ui.div(
                            "Joint prediction",
                            class_="joint-profile-title",
                        ),
                        ui.div(
                            joint,
                            class_="joint-profile-value",
                            style=f"color:{joint_color};",
                        ),
                    ),
                    ui.div(
                        "Patient point is positioned using the two locked "
                        "decision thresholds.",
                        class_="section-note",
                        style="max-width:300px; margin:0; text-align:right;",
                    ),
                    class_="joint-summary",
                ),
                joint_map(pe, pi),
                class_="joint-card",
            ),

        )


app = App(app_ui, server)
