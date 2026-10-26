"""Semi-synthetic data-generating process (DGP) with known ground truth (P2-F04).

Generates synthetic treatment (relative price change in [-0.15, +0.15]) with observed
confounding, binary renewal outcomes via a logistic model, closed-form ground truth
metrics, and executes five automated self-tests to guarantee validity under ADR-013,
P2-D2, P2-D2a, and P2-D2b.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, overload

import numpy as np
import pandas as pd

from aegis.config.loader import get_config
from aegis.schemas.dgp import DGPManifest, SelfTestResult
from aegis.utils.exceptions import DGPVerificationError


def _compute_file_sha256(path: Path | str) -> str:
    """Computes SHA-256 hash of a file."""
    p = Path(path)
    if not p.is_file():
        return ""
    hasher = hashlib.sha256()
    with p.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


@overload
def sigmoid(z: float) -> float: ...


@overload
def sigmoid(z: np.ndarray) -> np.ndarray: ...


def sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable sigmoid function."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))


def compute_baseline_log_odds(
    df: pd.DataFrame,
    base: float = 1.4,
    driver_age_ge_25: float = 0.3,
    car_age_lt_10: float = 0.2,
) -> np.ndarray:
    """Computes baseline renewal log-odds a(X) from pre-treatment rating factors (OD-1).

    Args:
        df: DataFrame containing driver_age_band_ord and car_age_band_ord.
        base: Baseline intercept.
        driver_age_ge_25: Increment for drivers aged 25+ (driver_age_band_ord >= 1).
        car_age_lt_10: Increment for vehicles aged < 10 (car_age_band_ord <= 1).

    Returns:
        np.ndarray: Vector of log-odds a(X_i).
    """
    age_ge_25 = (df["driver_age_band_ord"].to_numpy() >= 1).astype(float)
    car_lt_10 = (df["car_age_band_ord"].to_numpy() <= 1).astype(float)
    return base + driver_age_ge_25 * age_ge_25 + car_age_lt_10 * car_lt_10


def compute_price_sensitivity_slope(
    df: pd.DataFrame,
    base: float = 3.5,
    driver_age_ord: float = -0.4,
    car_age_ord: float = 0.3,
    interaction: float = -0.1,
) -> np.ndarray:
    """Computes negative price sensitivity slope b(X) with interaction (OD-1).

    Args:
        df: DataFrame containing driver_age_band_ord and car_age_band_ord.
        base: Base slope scale.
        driver_age_ord: Linear coefficient for driver age ordinal.
        car_age_ord: Linear coefficient for car age ordinal.
        interaction: Interaction coefficient between driver age and car age.

    Returns:
        np.ndarray: Vector of negative slopes b(X_i).
    """
    d_ord = df["driver_age_band_ord"].to_numpy().astype(float)
    c_ord = df["car_age_band_ord"].to_numpy().astype(float)
    raw_scale = base + driver_age_ord * d_ord + car_age_ord * c_ord + interaction * d_ord * c_ord
    return -raw_scale


def assign_confounded_treatment(
    loss_cost: np.ndarray,
    rng: np.random.Generator,
    confounding_dial: float = 0.5,
    residual_std: float = 0.05,
    corridor_width: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """Assigns treatment with observed confounding from GLM fitted loss cost (OD-2, FR-2).

    Args:
        loss_cost: Array of fitted loss costs per policy.
        rng: Pinned numpy random generator.
        confounding_dial: Strength parameter gamma in [0.0, 1.0].
        residual_std: Standard deviation of residual Gaussian noise.
        corridor_width: Maximum allowed adjustment magnitude (default 0.15).

    Returns:
        tuple[np.ndarray, np.ndarray]: (assigned_treatment, confounded_mean_mu)
    """
    med = float(np.median(loss_cost))
    q25 = float(np.percentile(loss_cost, 25))
    q75 = float(np.percentile(loss_cost, 75))
    iqr = q75 - q25 if (q75 - q25) > 1e-6 else float(np.std(loss_cost))
    if iqr < 1e-6:
        iqr = 1.0

    std_loss = (loss_cost - med) / iqr
    mu_t = confounding_dial * 0.08 * np.tanh(std_loss)
    noise = rng.normal(0.0, residual_std, size=len(loss_cost))
    treatment = np.clip(mu_t + noise, -corridor_width, corridor_width)
    return treatment, mu_t


def run_dgp_self_tests(
    df: pd.DataFrame,
    treatment: np.ndarray,
    mu_t: np.ndarray,
    q0: np.ndarray,
    tau: np.ndarray,
    epsilon: np.ndarray,
    a_x: np.ndarray,
    b_x: np.ndarray,
    corridor_width: float = 0.15,
    estimand_gap_tolerance: float = 0.02,
    overlap_floor_var: float = 0.001,
    dispersion_min_std: float = 0.010,
    homogeneity_max_corr: float = 0.05,
    elasticity_bound_product: float = 1.0,
) -> dict[str, SelfTestResult]:
    """Executes the five automated DGP self-tests mandated by ADR-013 and P2-D2a/b.

    Returns:
        dict[str, SelfTestResult]: Structured results for each self-test.

    Raises:
        DGPVerificationError: If any self-test fails.
    """
    results: dict[str, SelfTestResult] = {}

    # 1. Estimand Gap Test (FR-7)
    # Average slope over [-corridor_width, +corridor_width]:
    p_high = sigmoid(a_x + b_x * corridor_width)
    p_low = sigmoid(a_x - b_x * corridor_width)
    avg_slope = (p_high - p_low) / (2.0 * corridor_width)

    gap_data = pd.DataFrame(
        {
            "segment_id": df["segment_id"],
            "tau": tau,
            "avg_slope": avg_slope,
            "q0": q0,
        }
    )
    seg_grouped = gap_data.groupby("segment_id")
    sum_tau = seg_grouped["tau"].sum().to_numpy(dtype=float)
    sum_avg_slope = seg_grouped["avg_slope"].sum().to_numpy(dtype=float)
    sum_q0 = seg_grouped["q0"].sum().to_numpy(dtype=float)
    seg_ids = list(seg_grouped.groups.keys())

    truth_target = sum_tau / sum_q0
    avg_target = sum_avg_slope / sum_q0
    seg_gaps = np.abs(truth_target - avg_target)
    max_gap = float(np.max(seg_gaps))
    worst_idx = int(np.argmax(seg_gaps))
    worst_segment = str(seg_ids[worst_idx])

    gap_passed = max_gap <= estimand_gap_tolerance
    results["estimand_gap"] = SelfTestResult(
        name="estimand_gap",
        passed=gap_passed,
        observed_value=max_gap,
        threshold=estimand_gap_tolerance,
        details={"max_segment_gap": max_gap, "worst_segment": worst_segment},
    )
    if not gap_passed:
        raise DGPVerificationError(
            f"Estimand gap self-test failed: max gap {max_gap:.5f} exceeds tolerance "
            f"{estimand_gap_tolerance}",
            details={"max_gap": max_gap, "threshold": estimand_gap_tolerance},
        )

    # 2. Overlap Residual Variance Test (FR-8)
    res_var = float(np.var(treatment - mu_t))
    overlap_passed = res_var >= overlap_floor_var
    results["overlap"] = SelfTestResult(
        name="overlap",
        passed=overlap_passed,
        observed_value=res_var,
        threshold=overlap_floor_var,
        details={"residual_variance": res_var},
    )
    if not overlap_passed:
        raise DGPVerificationError(
            f"Overlap self-test failed: residual variance {res_var:.6f} below floor "
            f"{overlap_floor_var}",
            details={"residual_variance": res_var, "threshold": overlap_floor_var},
        )

    # 3. Between-Segment Dispersion Test (FR-9)
    seg_tau_mean = seg_grouped["tau"].mean()
    dispersion_std = float(seg_tau_mean.std())
    dispersion_passed = dispersion_std >= dispersion_min_std
    results["dispersion"] = SelfTestResult(
        name="dispersion",
        passed=dispersion_passed,
        observed_value=dispersion_std,
        threshold=dispersion_min_std,
        details={"between_segment_tau_std": dispersion_std},
    )
    if not dispersion_passed:
        raise DGPVerificationError(
            f"Dispersion self-test failed: between-segment tau std {dispersion_std:.5f} "
            f"below minimum {dispersion_min_std}",
            details={"observed_std": dispersion_std, "threshold": dispersion_min_std},
        )

    # 4. Within-Segment Homogeneity Test (FR-10, ADR-011)
    homo_data = pd.DataFrame(
        {
            "segment_id": df["segment_id"],
            "epsilon": epsilon,
            "fitted_loss_cost": df["fitted_loss_cost"],
        }
    )
    max_corr = 0.0
    worst_homo_seg = ""
    for seg_id, group in homo_data.groupby("segment_id"):
        eps_vals = group["epsilon"].to_numpy()
        loss_vals = group["fitted_loss_cost"].to_numpy()
        # If epsilon is constant within segment, correlation is zero by construction
        if np.std(eps_vals) < 1e-8 or np.std(loss_vals) < 1e-8:
            corr = 0.0
        else:
            r = np.corrcoef(eps_vals, loss_vals)[0, 1]
            corr = float(np.abs(r)) if np.isfinite(r) else 0.0

        if corr > max_corr:
            max_corr = corr
            worst_homo_seg = str(seg_id)

    homo_passed = max_corr <= homogeneity_max_corr
    results["homogeneity"] = SelfTestResult(
        name="homogeneity",
        passed=homo_passed,
        observed_value=max_corr,
        threshold=homogeneity_max_corr,
        details={"max_within_segment_corr": max_corr, "worst_segment": worst_homo_seg},
    )
    if not homo_passed:
        raise DGPVerificationError(
            f"Homogeneity self-test failed: max within-segment corr {max_corr:.4f} "
            f"exceeds tolerance {homogeneity_max_corr}",
            details={"max_corr": max_corr, "threshold": homogeneity_max_corr},
        )

    # 5. Elasticity Bound Test (FR-11, FR-5)
    max_bound_prod = float(np.max(np.abs(epsilon)) * corridor_width)
    bound_passed = max_bound_prod < elasticity_bound_product
    results["bound_guarantee"] = SelfTestResult(
        name="bound_guarantee",
        passed=bound_passed,
        observed_value=max_bound_prod,
        threshold=elasticity_bound_product,
        details={"max_epsilon_times_corridor": max_bound_prod},
    )
    if not bound_passed:
        raise DGPVerificationError(
            f"Elasticity bound self-test failed: max product {max_bound_prod:.4f} >= "
            f"{elasticity_bound_product}",
            details={"observed_product": max_bound_prod, "threshold": elasticity_bound_product},
        )

    return results


def generate_semi_synthetic_dataset(
    forest_features_path: Path | str,
    segments_path: Path | str,
    glm_loss_path: Path | str,
    config: Any | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, DGPManifest]:
    """Generates semi-synthetic training and truth sidecar datasets (FR-1..FR-12).

    Args:
        forest_features_path: Path to Stage 2 features_forest.csv.
        segments_path: Path to Stage 4 policy_segments.csv.
        glm_loss_path: Path to Stage 3 glm_fitted_loss_cost.csv.
        config: AEGISConfig instance. If None, loaded via get_config().

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, DGPManifest]:
            - training_df: Estimator training table (pre-treatment features, T, Y, segment_id)
            - truth_df: Isolated truth sidecar table (q0, tau, epsilon, segment_id)
            - manifest: Versioned DGPManifest artifact

    Raises:
        DGPVerificationError: If input alignment or self-tests fail.
    """
    if config is None:
        config = get_config()

    dgp_cfg = config.tier1_ml.synthetic_dgp

    # 1. Ingest input datasets
    forest_df = pd.read_csv(forest_features_path)
    seg_df = pd.read_csv(segments_path)
    glm_df = pd.read_csv(glm_loss_path)

    # Validate row counts and policy alignment
    n_expected = 413169
    if len(forest_df) != n_expected or len(seg_df) != n_expected or len(glm_df) != n_expected:
        raise DGPVerificationError(
            f"Input row counts mismatch: forest={len(forest_df)}, seg={len(seg_df)}, "
            f"glm={len(glm_df)} (expected {n_expected})",
            details={
                "forest_rows": len(forest_df),
                "seg_rows": len(seg_df),
                "glm_rows": len(glm_df),
            },
        )

    # Ensure deterministic sort by PolicyID
    forest_df = forest_df.sort_values("PolicyID").reset_index(drop=True)
    seg_df = seg_df.sort_values("PolicyID").reset_index(drop=True)
    glm_df = glm_df.sort_values("PolicyID").reset_index(drop=True)

    ids_aligned = (
        forest_df["PolicyID"].equals(seg_df["PolicyID"])
        and forest_df["PolicyID"].equals(glm_df["PolicyID"])
    )
    if not ids_aligned:
        raise DGPVerificationError("PolicyID alignment mismatch across input tables")

    # Combine needed fields for generation
    merged = forest_df.copy()
    merged["segment_id"] = seg_df["segment_id"]
    loss_col = (
        "fitted_loss_cost_per_year"
        if "fitted_loss_cost_per_year" in glm_df.columns
        else "fitted_loss_cost"
    )
    merged["fitted_loss_cost"] = glm_df[loss_col]

    # 2. Treatment assignment
    rng = np.random.default_rng(dgp_cfg.random_seed)
    treatment, mu_t = assign_confounded_treatment(
        loss_cost=merged["fitted_loss_cost"].to_numpy(),
        rng=rng,
        confounding_dial=dgp_cfg.confounding_dial,
        residual_std=dgp_cfg.residual_std,
        corridor_width=dgp_cfg.corridor_width,
    )

    # 3. Outcome & Ground Truth Generation
    a_x = compute_baseline_log_odds(
        merged,
        base=dgp_cfg.a_coeffs.base,
        driver_age_ge_25=dgp_cfg.a_coeffs.driver_age_ge_25,
        car_age_lt_10=dgp_cfg.a_coeffs.car_age_lt_10,
    )
    b_x = compute_price_sensitivity_slope(
        merged,
        base=dgp_cfg.b_coeffs.base,
        driver_age_ord=dgp_cfg.b_coeffs.driver_age_ord,
        car_age_ord=dgp_cfg.b_coeffs.car_age_ord,
        interaction=dgp_cfg.b_coeffs.interaction,
    )

    p_renew = sigmoid(a_x + b_x * treatment)
    renewed = (rng.uniform(0.0, 1.0, size=len(merged)) < p_renew).astype(int)

    # Ground truth definitions at Delta = 0
    q0 = np.asarray(sigmoid(a_x), dtype=float)
    tau = q0 * (1.0 - q0) * b_x
    epsilon = (1.0 - q0) * b_x

    # 4. Execute five automated self-tests
    self_test_results = run_dgp_self_tests(
        df=merged,
        treatment=treatment,
        mu_t=mu_t,
        q0=q0,
        tau=tau,
        epsilon=epsilon,
        a_x=a_x,
        b_x=b_x,
        corridor_width=dgp_cfg.corridor_width,
        estimand_gap_tolerance=dgp_cfg.estimand_gap_tolerance,
        overlap_floor_var=dgp_cfg.overlap_floor_var,
        dispersion_min_std=dgp_cfg.dispersion_min_std,
        homogeneity_max_corr=dgp_cfg.homogeneity_max_corr,
        elasticity_bound_product=dgp_cfg.elasticity_bound_product,
    )

    # 5. Build Training Table (strictly pre-treatment features, T, Y, segment_id; drop exposure)
    feature_cols = [c for c in forest_df.columns if c not in ("PolicyID", "exposure_capped")]
    training_cols = ["PolicyID"] + feature_cols + ["treatment_rate_change", "renewed", "segment_id"]

    training_df = pd.DataFrame(index=forest_df.index)
    training_df["PolicyID"] = forest_df["PolicyID"]
    for c in feature_cols:
        training_df[c] = forest_df[c]
    training_df["treatment_rate_change"] = np.round(treatment, 6)
    training_df["renewed"] = renewed
    training_df["segment_id"] = seg_df["segment_id"]
    training_df = pd.DataFrame(training_df[training_cols])

    # 6. Build Truth Sidecar Table
    truth_df = pd.DataFrame(
        {
            "PolicyID": forest_df["PolicyID"],
            "baseline_renewal_q0": np.round(q0, 6),
            "true_tau": np.round(tau, 6),
            "true_epsilon": np.round(epsilon, 6),
            "segment_id": seg_df["segment_id"],
        }
    )

    # 7. Build Manifest
    input_hashes = {
        "features_forest": _compute_file_sha256(forest_features_path),
        "policy_segments": _compute_file_sha256(segments_path),
        "glm_fitted_loss_cost": _compute_file_sha256(glm_loss_path),
    }

    raw_config = Path("params.yaml").read_bytes() if Path("params.yaml").is_file() else b""
    config_fingerprint = hashlib.sha256(raw_config).hexdigest()

    manifest = DGPManifest(
        random_seed=dgp_cfg.random_seed,
        sample_size=len(training_df),
        corridor_width=dgp_cfg.corridor_width,
        confounding_dial=dgp_cfg.confounding_dial,
        residual_std=dgp_cfg.residual_std,
        self_tests=self_test_results,
        input_hashes=input_hashes,
        config_fingerprint=config_fingerprint,
    )

    return training_df, truth_df, manifest


def run_synthetic_dgp_pipeline(config: Any | None = None) -> DGPManifest:
    """Orchestrates semi-synthetic DGP generation, self-tests, and persistence (FR-12).

    Returns:
        DGPManifest: The validated metadata manifest.
    """
    if config is None:
        config = get_config()

    forest_path = Path(config.tier1_ml.feature_paths.forest_view)
    seg_path = Path(config.tier1_ml.segment_paths.policy_segments)
    glm_path = Path(config.tier1_ml.glm_paths.fitted_loss_cost)

    dgp_paths = config.tier1_ml.dgp_paths
    train_out = Path(dgp_paths.training_table)
    truth_out = Path(dgp_paths.truth_sidecar)
    manifest_out = Path(dgp_paths.manifest)

    train_out.parent.mkdir(parents=True, exist_ok=True)
    truth_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)

    training_df, truth_df, manifest = generate_semi_synthetic_dataset(
        forest_features_path=forest_path,
        segments_path=seg_path,
        glm_loss_path=glm_path,
        config=config,
    )

    # Persist with deterministic newline and formatting
    training_df.to_csv(train_out, index=False, lineterminator="\n")
    truth_df.to_csv(truth_out, index=False, lineterminator="\n")
    manifest_out.write_text(
        json.dumps(manifest.model_dump(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return manifest
