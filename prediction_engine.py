# -*- coding: utf-8 -*-
"""
DTC dual-endpoint deployment prediction engine
================================================

This module reproduces the locked prediction logic for deployment.

Formal rules
------------
1. No retraining.
2. No new feature selection.
3. No hyperparameter tuning.
4. No threshold optimization.
5. Efficacy = mean probability from 5 locked CatBoost MI models.
6. Injury = mean probability from 5 locked Logistic Regression MI models.
7. Classification uses the locked Youden threshold calculated from
   mean Training OOF probabilities across the 5 MI datasets.
8. New-patient inputs must be complete for all required web fields.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config" / "deployment_config.json"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

EFFICACY_FEATURES = CONFIG["efficacy"]["features"]
INJURY_FEATURES = CONFIG["injury"]["features"]

EFFICACY_THRESHOLD = float(CONFIG["efficacy"]["locked_threshold"])
INJURY_THRESHOLD = float(CONFIG["injury"]["locked_threshold"])

RSM_T_HALF_MEAN_BY_MI = {
    int(k): float(v)
    for k, v in CONFIG["injury"][
        "rsm_t_half_training_uncensored_mean_by_mi"
    ].items()
}

SURGERY_TYPE_MAP = {
    "Total": 1,
    "Near-total": 2,
    "Subtotal": 3,
}
ATA2025_MAP = {
    "Low": 0,
    "Low-intermediate": 1,
    "Intermediate-high": 2,
    "High": 3,
}
VITC_MAP = {
    "Decreased": 0,
    "Increased": 1,
}


def _finite_number(name: str, value: Any) -> float:
    try:
        x = float(value)
    except Exception as e:
        raise ValueError(f"{name} must be numeric.") from e
    if not np.isfinite(x):
        raise ValueError(f"{name} must be finite.")
    return x


def _nonnegative(name: str, value: Any) -> float:
    x = _finite_number(name, value)
    if x < 0:
        raise ValueError(f"{name} cannot be negative.")
    return x


def _normalize_choice(value: Any, allowed: dict, name: str):
    if value in allowed:
        return value
    raise ValueError(
        f"{name} must be one of: {', '.join(map(str, allowed.keys()))}."
    )


def validate_raw_input(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize the raw web input.

    Required keys
    -------------
    first_rai_activity
    number_metastatic
    pre_rai_tg
    surgery_to_rai_days
    pre_rai_tgab
    surgery_type
    ata2025_risk
    max_ln_metastasis
    bmi
    lp_peak_count
    rp_peak_count
    lp_vitc_response
    rp_vitc_response
    rsm_t_half_status
    rsm_t_half
    rsm_ef

    For rsm_t_half_status == "Prolonged beyond observation window",
    rsm_t_half may be None because the continuous component is forced to 0.
    """
    required = [
        "first_rai_activity",
        "number_metastatic",
        "pre_rai_tg",
        "surgery_to_rai_days",
        "pre_rai_tgab",
        "surgery_type",
        "ata2025_risk",
        "max_ln_metastasis",
        "bmi",
        "lp_peak_count",
        "rp_peak_count",
        "lp_vitc_response",
        "rp_vitc_response",
        "rsm_t_half_status",
        "rsm_ef",
    ]

    missing = [k for k in required if k not in raw]
    if missing:
        raise ValueError(f"Missing required inputs: {missing}")

    x: Dict[str, Any] = {}

    x["first_rai_activity"] = _nonnegative(
        "First RAI activity", raw["first_rai_activity"]
    )

    nm = _nonnegative(
        "Number of metastatic lymph nodes", raw["number_metastatic"]
    )
    if not float(nm).is_integer():
        raise ValueError(
            "Number of metastatic lymph nodes must be an integer."
        )
    x["number_metastatic"] = int(nm)

    x["pre_rai_tg"] = _nonnegative("Pre-RAI Tg", raw["pre_rai_tg"])
    x["surgery_to_rai_days"] = _nonnegative(
        "Surgery-to-RAI interval", raw["surgery_to_rai_days"]
    )
    x["pre_rai_tgab"] = _nonnegative(
        "Pre-RAI TgAb", raw["pre_rai_tgab"]
    )

    x["surgery_type"] = _normalize_choice(
        raw["surgery_type"], SURGERY_TYPE_MAP, "Surgery type"
    )
    x["ata2025_risk"] = _normalize_choice(
        raw["ata2025_risk"], ATA2025_MAP, "2025 ATA recurrence risk"
    )

    x["max_ln_metastasis"] = _nonnegative(
        "Maximum metastatic lymph node size", raw["max_ln_metastasis"]
    )

    bmi = _finite_number("BMI", raw["bmi"])
    if bmi <= 0:
        raise ValueError("BMI must be > 0.")
    x["bmi"] = bmi

    x["lp_peak_count"] = _nonnegative(
        "Left parotid peak count", raw["lp_peak_count"]
    )
    x["rp_peak_count"] = _nonnegative(
        "Right parotid peak count", raw["rp_peak_count"]
    )

    x["lp_vitc_response"] = _normalize_choice(
        raw["lp_vitc_response"], VITC_MAP,
        "Left parotid response to vitamin C"
    )
    x["rp_vitc_response"] = _normalize_choice(
        raw["rp_vitc_response"], VITC_MAP,
        "Right parotid response to vitamin C"
    )

    allowed_status = {
        "Measurable": 0,
        "Prolonged beyond observation window": 1,
    }
    x["rsm_t_half_status"] = _normalize_choice(
        raw["rsm_t_half_status"], allowed_status,
        "Right submandibular gland T1/2 status"
    )

    if x["rsm_t_half_status"] == "Measurable":
        if "rsm_t_half" not in raw or raw["rsm_t_half"] is None:
            raise ValueError(
                "Right submandibular gland T1/2 is required when measurable."
            )
        x["rsm_t_half"] = _nonnegative(
            "Right submandibular gland T1/2", raw["rsm_t_half"]
        )
    else:
        # The deployment transformation sets the centered continuous
        # component to zero for censored observations.
        x["rsm_t_half"] = None

    ef = _finite_number(
        "Right submandibular gland EF", raw["rsm_ef"]
    )
    if ef < 0 or ef > 100:
        raise ValueError(
            "Right submandibular gland EF must be between 0 and 100 (%)."
        )
    x["rsm_ef"] = ef

    return x


def build_efficacy_row(raw: Dict[str, Any]) -> pd.DataFrame:
    s = validate_raw_input(raw)

    surgery_code = SURGERY_TYPE_MAP[s["surgery_type"]]

    row = {
        "first_rai_activity": s["first_rai_activity"],
        "number_metastatic": s["number_metastatic"],
        "pre_rai_tg": s["pre_rai_tg"],
        "surgery_to_rai_days": s["surgery_to_rai_days"],
        "pre_rai_tgab": s["pre_rai_tgab"],
        "surgery_type__2": 1.0 if surgery_code == 2 else 0.0,
        "surgery_type__3": 1.0 if surgery_code == 3 else 0.0,
        "ata2025_risk": float(ATA2025_MAP[s["ata2025_risk"]]),
        "max_ln_metastasis": s["max_ln_metastasis"],
    }

    return pd.DataFrame(
        [[row[c] for c in EFFICACY_FEATURES]],
        columns=EFFICACY_FEATURES,
        dtype=float,
    )


def build_injury_row_for_mi(
    raw: Dict[str, Any],
    imputation_id: int,
) -> pd.DataFrame:
    s = validate_raw_input(raw)

    if imputation_id not in RSM_T_HALF_MEAN_BY_MI:
        raise ValueError(f"Unknown imputation_id: {imputation_id}")

    parotid_peak_mean = (
        s["lp_peak_count"] + s["rp_peak_count"]
    ) / 2.0

    parotid_vitc_response_count = (
        VITC_MAP[s["lp_vitc_response"]]
        + VITC_MAP[s["rp_vitc_response"]]
    )

    censored = int(
        s["rsm_t_half_status"]
        == "Prolonged beyond observation window"
    )

    if censored == 1:
        centered = 0.0
    else:
        centered = (
            float(s["rsm_t_half"])
            - RSM_T_HALF_MEAN_BY_MI[imputation_id]
        )

    row = {
        "first_rai_activity": s["first_rai_activity"],
        "bmi": s["bmi"],
        "parotid_peak_mean": parotid_peak_mean,
        "parotid_vitc_response_count":
            float(parotid_vitc_response_count),
        "rsm_t_half_uncensored_centered": centered,
        "rsm_t_half_censored": float(censored),
        "rsm_ef": s["rsm_ef"],
    }

    return pd.DataFrame(
        [[row[c] for c in INJURY_FEATURES]],
        columns=INJURY_FEATURES,
        dtype=float,
    )


def _load_models():
    efficacy_models = {}
    injury_models = {}

    for m in range(1, 6):
        ep = (
            BASE_DIR / "models" / "efficacy"
            / f"M{m:02d}_CatBoost.joblib"
        )
        ip = (
            BASE_DIR / "models" / "injury"
            / f"M{m:02d}_Logistic_Regression.joblib"
        )

        if not ep.exists():
            raise FileNotFoundError(ep)
        if not ip.exists():
            raise FileNotFoundError(ip)

        efficacy_models[m] = joblib.load(ep)
        injury_models[m] = joblib.load(ip)

    return efficacy_models, injury_models


EFFICACY_MODELS, INJURY_MODELS = _load_models()



def _predict_positive_probability(model, X: pd.DataFrame) -> float:
    """
    Return P(class=1) while preserving the fitted preprocessing.

    This helper also avoids unnecessary sklearn Pipeline metadata/tag
    validation for a pipeline that contains only a fitted classifier.
    """
    if hasattr(model, "named_steps"):
        steps = model.named_steps

        if list(steps.keys()) == ["clf"]:
            p = steps["clf"].predict_proba(X)
        elif "scaler" in steps and "clf" in steps:
            X_scaled = steps["scaler"].transform(X)
            p = steps["clf"].predict_proba(X_scaled)
        else:
            p = model.predict_proba(X)
    else:
        p = model.predict_proba(X)

    p = np.asarray(p)
    if p.shape != (1, 2):
        raise RuntimeError(
            f"Invalid predict_proba shape: {p.shape}; expected (1, 2)."
        )
    return float(p[0, 1])


def model_integrity_qc() -> Dict[str, Any]:
    """
    Confirm that all 10 locked models accept the expected input structure.
    """
    rows = []

    for m, model in EFFICACY_MODELS.items():
        clf = (
            model.named_steps["clf"]
            if hasattr(model, "named_steps")
            else model
        )
        raw_feature_names = getattr(
            clf, "feature_names_", None
        )
        feature_names = (
            list(raw_feature_names)
            if raw_feature_names is not None
            else []
        )
        passed = (
            not feature_names
            or feature_names == EFFICACY_FEATURES
        )
        rows.append({
            "endpoint": "efficacy",
            "imputation_id": m,
            "model": "CatBoost",
            "feature_n": len(EFFICACY_FEATURES),
            "feature_names_match": bool(passed),
        })

    for m, model in INJURY_MODELS.items():
        raw_feature_names = getattr(
            model, "feature_names_in_", None
        )
        feature_names = (
            list(raw_feature_names)
            if raw_feature_names is not None
            else []
        )
        passed = (
            not feature_names
            or feature_names == INJURY_FEATURES
        )
        rows.append({
            "endpoint": "injury",
            "imputation_id": m,
            "model": "Logistic Regression",
            "feature_n": len(INJURY_FEATURES),
            "feature_names_match": bool(passed),
        })

    overall = all(r["feature_names_match"] for r in rows)

    return {
        "passed": overall,
        "details": rows,
        "efficacy_threshold": EFFICACY_THRESHOLD,
        "injury_threshold": INJURY_THRESHOLD,
    }


def predict_patient(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict both endpoints for one complete new patient.

    Returns patient-level mean probabilities across 5 MI models.
    """
    # Validate once before prediction
    normalized = validate_raw_input(raw)

    # ---- efficacy ----
    X_eff = build_efficacy_row(normalized)
    efficacy_mi_prob = {}

    for m, model in EFFICACY_MODELS.items():
        efficacy_mi_prob[m] = _predict_positive_probability(
            model, X_eff
        )

    efficacy_mean_prob = float(
        np.mean(list(efficacy_mi_prob.values()))
    )
    efficacy_pred = (
        "Non-ER"
        if efficacy_mean_prob >= EFFICACY_THRESHOLD
        else "ER"
    )

    # ---- injury ----
    injury_mi_prob = {}
    injury_derived_by_mi = {}

    for m, model in INJURY_MODELS.items():
        X_inj = build_injury_row_for_mi(normalized, m)
        injury_mi_prob[m] = _predict_positive_probability(
            model, X_inj
        )
        injury_derived_by_mi[m] = {
            "rsm_t_half_uncensored_centered":
                float(X_inj.iloc[0][
                    "rsm_t_half_uncensored_centered"
                ]),
            "rsm_t_half_censored":
                int(X_inj.iloc[0]["rsm_t_half_censored"]),
        }

    injury_mean_prob = float(
        np.mean(list(injury_mi_prob.values()))
    )
    injury_pred = (
        "Injury"
        if injury_mean_prob >= INJURY_THRESHOLD
        else "No injury"
    )

    joint_profile = f"{efficacy_pred} + {injury_pred}"

    return {
        "efficacy": {
            "positive_class": "Non-ER",
            "mi_probabilities": {
                f"M{k:02d}": v
                for k, v in efficacy_mi_prob.items()
            },
            "mean_probability": efficacy_mean_prob,
            "locked_threshold": EFFICACY_THRESHOLD,
            "predicted_class": efficacy_pred,
        },
        "injury": {
            "positive_class": "Injury",
            "mi_probabilities": {
                f"M{k:02d}": v
                for k, v in injury_mi_prob.items()
            },
            "mean_probability": injury_mean_prob,
            "locked_threshold": INJURY_THRESHOLD,
            "predicted_class": injury_pred,
        },
        "joint_profile": joint_profile,
        "deployment_audit": {
            "parotid_peak_mean":
                (
                    normalized["lp_peak_count"]
                    + normalized["rp_peak_count"]
                ) / 2.0,
            "parotid_vitc_response_count":
                (
                    VITC_MAP[normalized["lp_vitc_response"]]
                    + VITC_MAP[normalized["rp_vitc_response"]]
                ),
            "injury_t_half_by_mi": injury_derived_by_mi,
        },
    }


if __name__ == "__main__":
    qc = model_integrity_qc()
    print("=" * 72)
    print("DTC deployment prediction engine")
    print("=" * 72)
    print(f"Model integrity QC passed: {qc['passed']}")
    print(f"Efficacy locked threshold: {EFFICACY_THRESHOLD:.12f}")
    print(f"Injury locked threshold:   {INJURY_THRESHOLD:.12f}")
    print()
    for r in qc["details"]:
        print(
            f"{r['endpoint']:8s} M{r['imputation_id']:02d} "
            f"{r['model']:20s} "
            f"features={r['feature_n']} "
            f"match={r['feature_names_match']}"
        )
    print()
    print(
        "No patient prediction was performed. "
        "Next QC step: compare deployment predictions against "
        "saved Step12/Step13 patient-level probabilities."
    )
