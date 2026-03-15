"""
run_pipeline.py — Phases 1 + 2 entry point.

Usage
-----
    python run_pipeline.py               # full pipeline
    python run_pipeline.py --no-smote    # skip SMOTE
    python run_pipeline.py --skip-phase1 # skip data loading if already processed
"""
import argparse
import sys
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.loader import load_raw_data, get_class_distribution
from src.data.preprocessor import preprocess
from src.features.engineer import engineer_features, get_feature_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fraud Detection — Pipeline")
    parser.add_argument("--no-smote",    action="store_true", help="Skip SMOTE")
    parser.add_argument("--skip-phase1", action="store_true", help="Skip loading if splits exist")
    parser.add_argument("--data-path",   type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("  FRAUD DETECTION SYSTEM — Pipeline")
    logger.info("=" * 60)

    # ── Phase 1: Load + Preprocess ────────────────────────────
    from src.config import DATA_RAW_PATH, DATA_PROCESSED_DIR
    splits_path = DATA_PROCESSED_DIR / "splits.pkl"

    if args.skip_phase1 and splits_path.exists():
        logger.info("Skipping Phase 1 — using existing splits")
        from src.data.preprocessor import load_splits
        X_train, X_test, y_train, y_test = load_splits()
    else:
        logger.info("Phase 1: Data Pipeline")
        data_path = Path(args.data_path) if args.data_path else DATA_RAW_PATH
        df_raw = load_raw_data(data_path)
        dist   = get_class_distribution(df_raw)
        logger.info(f"Class distribution: {dist}")

        X_train, X_test, y_train, y_test = preprocess(
            df_raw,
            apply_smote=not args.no_smote,
            save_artifacts=True,
        )

    # ── Phase 2: Feature Engineering ──────────────────────────
    logger.info("Phase 2: Feature Engineering")
    from src.config import DATA_RAW_PATH
    df_raw        = load_raw_data(DATA_RAW_PATH)
    df_engineered = engineer_features(df_raw, save=True)
    feature_names = get_feature_names()

    # ── Summary ───────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  PIPELINE COMPLETE")
    logger.info(f"  X_train shape       : {X_train.shape}")
    logger.info(f"  X_test  shape       : {X_test.shape}")
    logger.info(f"  Engineered features : {len(feature_names)}")
    logger.info(f"  y_train fraud       : {y_train.sum():,} / {len(y_train):,}")
    logger.info(f"  y_test  fraud       : {y_test.sum():,}  / {len(y_test):,}")
    logger.info("=" * 60)
    logger.success("Ready for Phase 3: Model Training 🚀")


if __name__ == "__main__":
    main()
