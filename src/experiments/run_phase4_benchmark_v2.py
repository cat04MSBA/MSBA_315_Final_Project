"""Phase 4 benchmark on the v2 corpus.

Same harness as the canonical Phase 4 benchmark
(``src.experiments.run_phase4_benchmark``) but driven by v2 inputs:

* train data from ``data/processed/v2/films_joined_v2.parquet`` and
  ``split_assignments_v2.parquet``
* feature matrices from the v2 per-group parquets in
  ``data/processed/v2/``
* outputs to ``reports/tables/phase4_benchmark_v2.csv``,
  ``runs/phase_4_v2/``, and ``data/processed/v2/phase4_primary_model_<target>_v2.joblib``

The v1 phase4 source code in ``src/models/phase4/`` is reused without
modification. Path-redirection happens by monkey-patching the
load-and-config seams at the top of this module before any benchmark
function is called.

Modes::

    python -m src.experiments.run_phase4_benchmark_v2 --mode primary
    python -m src.experiments.run_phase4_benchmark_v2 --mode mpnet
    python -m src.experiments.run_phase4_benchmark_v2 --mode full
"""

from __future__ import annotations

# Allow running by file path; no-op under `python -m`.
if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import warnings
from dataclasses import replace
from pathlib import Path

import joblib
import pandas as pd

# Silence sklearn 1.8 penalty deprecation noise (matches canonical script).
warnings.filterwarnings(
    "ignore",
    message=".*penalty.*was deprecated in version 1.8.*",
    category=FutureWarning,
)

from src.features.baseline_features import BaselineFeatureConfig
from src.features.targets import add_targets
from src.models.phase4 import benchmark as bm
from src.models.phase4.benchmark import BenchmarkConfig, run_full_benchmark
from src.models.phase4.families import FAMILIES
from src.models.phase4.matrices import MATRICES, MatrixSpec
from src.utils import paths
from src.utils.logging import get_logger, set_log_level

logger = get_logger(__name__)


V2_DIR = paths.DATA_PROCESSED_DIR / "v2"


# ---------------------------------------------------------------------------
# v2 monkey-patches
# ---------------------------------------------------------------------------

def _v2_baseline_config(use_mpnet: bool, drop_lexical_sentiment: bool) -> BaselineFeatureConfig:
    """BaselineFeatureConfig with all paths pointing to v2 parquets."""
    embedding_path = (
        V2_DIR / "features_embedding_mpnet_v2.parquet"
        if use_mpnet
        else V2_DIR / "features_embedding_v2.parquet"
    )
    return BaselineFeatureConfig(
        include_log_budget=False,
        log_transform_structural=True,
        include_log_runtime=True,
        include_lexical=not drop_lexical_sentiment,
        lexical_features_path=V2_DIR / "features_lexical_v2.parquet",
        include_sentiment=not drop_lexical_sentiment,
        sentiment_features_path=V2_DIR / "features_sentiment_v2.parquet",
        include_topic=True,
        topic_features_path=V2_DIR / "features_topic_v2.parquet",
        include_character_network=True,
        character_network_features_path=V2_DIR / "features_character_network_v2.parquet",
        include_embedding=True,
        embedding_features_path=embedding_path,
    )


def _patch_v2_matrices() -> None:
    """Replace the four MATRICES entries with v2-path versions in-place."""
    MATRICES["all_five"] = MatrixSpec(
        name="all_five",
        feature_config=_v2_baseline_config(use_mpnet=False, drop_lexical_sentiment=False),
        description="v2 corpus: structural + lexical + sentiment + topic + character_network + MiniLM embeddings",
    )
    MATRICES["standalone_positive_union"] = MatrixSpec(
        name="standalone_positive_union",
        feature_config=_v2_baseline_config(use_mpnet=False, drop_lexical_sentiment=True),
        description="v2 corpus: structural + topic + character_network + MiniLM embeddings (drops lexical & sentiment)",
    )
    MATRICES["all_five_mpnet"] = MatrixSpec(
        name="all_five_mpnet",
        feature_config=_v2_baseline_config(use_mpnet=True, drop_lexical_sentiment=False),
        description="v2 corpus: structural + lexical + sentiment + topic + character_network + mpnet embeddings",
    )
    MATRICES["standalone_positive_union_mpnet"] = MatrixSpec(
        name="standalone_positive_union_mpnet",
        feature_config=_v2_baseline_config(use_mpnet=True, drop_lexical_sentiment=True),
        description="v2 corpus: structural + topic + character_network + mpnet embeddings",
    )
    logger.info("Patched MATRICES with v2 versions")


def _v2_load_train_split() -> pd.DataFrame:
    """v2 replacement for benchmark.load_train_split."""
    df_full = pd.read_parquet(V2_DIR / "films_joined_v2.parquet")
    splits = pd.read_parquet(V2_DIR / "split_assignments_v2.parquet")
    df_full = add_targets(df_full)
    train_ids = splits.loc[splits["split"] == "train", "imdb_id"]
    df_train = df_full[df_full["imdb_id"].isin(train_ids)].reset_index(drop=True)
    logger.info("v2 train split: %d films", len(df_train))
    return df_train


def _v2_save_winner_artifact(target, matrix_name, family_name, cv_result, pattern):
    """v2 replacement that writes to data/processed/v2/ instead of data/processed/."""
    out = V2_DIR / pattern.format(target=target)
    bundle = {
        "target": target,
        "matrix": matrix_name,
        "family": family_name,
        "best_params": cv_result.best_params,
        "oof_metrics_global": cv_result.oof_metrics_global,
        "estimator": cv_result.fitted_estimator,
    }
    joblib.dump(bundle, out)
    logger.info("Saved v2 Phase 4 winner for %s: %s", target, out)
    return out


# Phase ID for save_run; redirects runs/phase_4/ → runs/phase_4_v2/.
_V2_PHASE = "phase_4_v2"


def _patch_evaluate_cell_phase() -> None:
    """Wrap evaluate_cell to route save_run output to runs/phase_4_v2/."""
    from src.experiments import save_run as save_run_mod
    original_save_run = save_run_mod.save_run

    def _v2_save_run(*args, **kwargs):
        # Override the phase argument to force runs/phase_4_v2/.
        if "phase" in kwargs:
            kwargs["phase"] = _V2_PHASE
        elif args:
            args = (_V2_PHASE,) + tuple(args[1:])
        return original_save_run(*args, **kwargs)

    save_run_mod.save_run = _v2_save_run
    bm.save_run = _v2_save_run  # bm imported save_run by name
    logger.info("Patched save_run to route to runs/%s/", _V2_PHASE)


def _apply_all_v2_patches() -> None:
    """Idempotent: applies all monkey-patches needed for v2 paths."""
    _patch_v2_matrices()
    bm.load_train_split = _v2_load_train_split
    bm.save_winner_artifact = _v2_save_winner_artifact
    _patch_evaluate_cell_phase()


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------


def _primary_v2_config() -> BenchmarkConfig:
    """v2 primary tier — the 6-family roster (linear/HistGB/RF/SVM-RBF + LightGBM + XGBoost) on both MiniLM matrices."""
    return BenchmarkConfig(
        mode="full",
        families=tuple(f.name for f in FAMILIES.values() if f.tier == "primary"),
        secondary_families=(),
        matrices=("all_five", "standalone_positive_union"),
        secondary_matrices=(),
        out_table=paths.REPORTS_TABLES_DIR / "phase4_benchmark_v2.csv",
        out_paired_table=paths.REPORTS_TABLES_DIR / "phase4_paired_tests_v2.csv",
        model_artifact_pattern="phase4_primary_model_{target}_v2.joblib",
        save_models=True,
    )


def _mpnet_v2_config() -> BenchmarkConfig:
    """v2 primary tier on mpnet matrices."""
    return BenchmarkConfig(
        mode="full",
        families=tuple(f.name for f in FAMILIES.values() if f.tier == "primary"),
        secondary_families=(),
        matrices=("all_five_mpnet", "standalone_positive_union_mpnet"),
        secondary_matrices=(),
        out_table=paths.REPORTS_TABLES_DIR / "phase4_benchmark_mpnet_v2.csv",
        out_paired_table=paths.REPORTS_TABLES_DIR / "phase4_paired_tests_mpnet_v2.csv",
        model_artifact_pattern="phase4_primary_model_{target}_v2.joblib",
        save_models=True,
    )


def _smoke_v2_config() -> BenchmarkConfig:
    """v2 smoke: linear only, all_five only, roi_gt_2 only."""
    smoke_linear = replace(
        FAMILIES["linear"],
        regression_grid={"alpha": [0.1, 1.0, 10.0]},
        classification_grid={"C": [0.1, 1.0, 10.0]},
    )
    FAMILIES[smoke_linear.name] = smoke_linear
    return BenchmarkConfig(
        mode="smoke",
        families=("linear",),
        secondary_families=(),
        matrices=("all_five",),
        secondary_matrices=(),
        targets=("roi_gt_2",),
        save_models=False,
        out_table=paths.REPORTS_TABLES_DIR / "phase4_smoke_v2.csv",
        out_paired_table=paths.REPORTS_TABLES_DIR / "phase4_smoke_paired_v2.csv",
    )


CONFIG_BUILDERS = {
    "smoke": _smoke_v2_config,
    "primary": _primary_v2_config,
    "mpnet": _mpnet_v2_config,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=sorted(CONFIG_BUILDERS), default="primary")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    set_log_level(args.log_level)
    _apply_all_v2_patches()

    cfg = CONFIG_BUILDERS[args.mode]()
    cfg = replace(cfg, n_jobs=args.n_jobs)

    logger.info("Phase 4 v2 benchmark starting: mode=%s", args.mode)
    logger.info("Config: families=%s matrices=%s targets=%s",
                cfg.families, cfg.matrices, cfg.targets)
    result = run_full_benchmark(cfg)
    logger.info("Phase 4 v2 benchmark complete.")
    if result["winners"]:
        for target, winfo in result["winners"].items():
            logger.info(
                "  Winner %s: %s on %s (OOF=%s)",
                target, winfo["family"], winfo["matrix"],
                winfo["oof_metrics_global"],
            )


if __name__ == "__main__":
    main()
