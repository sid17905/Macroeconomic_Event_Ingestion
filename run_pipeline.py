"""One-command entry point for the CrowdWisdomTrading macro-event pipeline.

Examples
--------
    python run_pipeline.py --stage all
    python run_pipeline.py --stage ingest
    python run_pipeline.py --stage trades features model evaluate report

Stages run in canonical order regardless of the order given. The SQLite
schema is (re)initialised on every invocation, and each stage is idempotent,
so the pipeline can be re-run freely (it reads cached raw data from data/raw).
"""
from __future__ import annotations

import argparse
import logging
import time

from src.config import load_config
from src.db.database import connect, init_db
from src.logging_utils import configure_logging, get_logger

STAGES = ["ingest", "trades", "features", "model", "evaluate", "report"]


def _parse_args():
    p = argparse.ArgumentParser(
        description="CrowdWisdomTrading macro-event ML pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--stage", nargs="+", default=["all"], choices=["all", *STAGES],
        help="Stage(s) to run. Space-separated for several; 'all' runs everything.",
    )
    p.add_argument("--config", default=None, help="Path to config.yaml")
    p.add_argument("--log-level", default="INFO", help="Logging level (DEBUG/INFO/...)")
    return p.parse_args()


def _expand_stages(requested: list[str]) -> list[str]:
    if "all" in requested:
        return list(STAGES)
    return [s for s in STAGES if s in requested]  # canonical order


def run_stage(stage: str, cfg, conn) -> None:
    """Dispatch a single stage. Imports are lazy so --help needs no heavy deps."""
    if stage == "ingest":
        from src.ingestion import run_ingestion
        run_ingestion(cfg, conn)
    elif stage == "trades":
        from src.trades.backtest import run_trades
        run_trades(cfg, conn)
    elif stage == "features":
        from src.features.build_features import run_features
        run_features(cfg, conn)
    elif stage == "model":
        from src.models.train import run_walkforward
        run_walkforward(cfg, conn)
    elif stage == "evaluate":
        from src.evaluation.evaluate import run_evaluation
        run_evaluation(cfg, conn)
    elif stage == "report":
        from src.viz.report import run_reports
        run_reports(cfg, conn)
    else:  # pragma: no cover - guarded by argparse choices
        raise ValueError(f"Unknown stage: {stage}")


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    configure_logging(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        log_file=cfg.data_processed / "pipeline.log",
    )
    log = get_logger("pipeline")

    conn = connect(cfg.db_path)
    init_db(conn)  # always ensure schema exists

    stages = _expand_stages(args.stage)
    log.info("Running stages: %s", ", ".join(stages))

    try:
        for stage in stages:
            t0 = time.time()
            log.info("--- Stage: %-9s -------------------------------", stage)
            run_stage(stage, cfg, conn)
            log.info("Stage '%s' finished in %.1fs", stage, time.time() - t0)
    finally:
        conn.close()
    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()
