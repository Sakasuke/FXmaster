"""ローソク足データを取得してローカルにキャッシュする.

cTrader と OANDA 両方に対応しています。
config.yaml の broker.name で自動的に切り替わります。

Usage:
    # LTF (M5) と HTF (H1) を両方取得 (推奨)
    python scripts/fetch_data.py --instrument USD_JPY --days 365

    # 個別に1本だけ取得
    python scripts/fetch_data.py --instrument USD_JPY --granularity M5 --days 90
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fxmaster.config import load_config
from fxmaster.data.candles import save_candles_csv
from fxmaster.utils.logger import get_logger


# --------------------------------------------------------------------------
# OANDA パス（既存処理）
# --------------------------------------------------------------------------

def _run_oanda(args, cfg, logger) -> None:
    from fxmaster.data import OandaDataFetcher
    from fxmaster.data.fetcher import OandaCredentials

    instrument = args.instrument or cfg.trading.instrument
    granularities = (
        [args.granularity]
        if args.granularity
        else [cfg.trading.ltf_granularity, cfg.trading.htf_granularity]
    )

    creds = OandaCredentials(
        account_id=cfg.broker.account_id,
        access_token=cfg.broker.access_token,
        environment=cfg.broker.environment,
    )
    fetcher = OandaDataFetcher(creds)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.days)

    for gran in granularities:
        logger.info("OANDA: %s %s を取得中 (%d 日間)...", instrument, gran, args.days)
        df = fetcher.fetch_candles(instrument, gran, start, end)
        logger.info("  取得完了: %d 本", len(df))
        out_path = Path(cfg.data.cache_dir) / f"{instrument}_{gran}.csv"
        save_candles_csv(df, out_path)
        logger.info("  保存先: %s", out_path)


# --------------------------------------------------------------------------
# cTrader パス（Twisted ベース）
# --------------------------------------------------------------------------

def _run_ctrader(args, cfg, logger) -> None:
    """Twisted reactor を使って cTrader からデータを取得する."""
    from twisted.internet import reactor, defer
    from fxmaster.data.ctrader_fetcher import CTraderClient, CTraderCredentials

    instrument = args.instrument or cfg.trading.instrument
    granularities = (
        [args.granularity]
        if args.granularity
        else [cfg.trading.ltf_granularity, cfg.trading.htf_granularity]
    )
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.days)

    creds = CTraderCredentials(
        client_id=cfg.broker.client_id,
        client_secret=cfg.broker.client_secret,
        access_token=cfg.broker.access_token,
        account_id=int(cfg.broker.account_id),
        environment=cfg.broker.environment,
    )
    client = CTraderClient(creds)
    result: dict = {"ok": False, "error": None}

    @defer.inlineCallbacks
    def run():
        try:
            yield client.connect_and_auth()

            for gran in granularities:
                logger.info(
                    "cTrader: %s %s を取得中 (%d 日間)...", instrument, gran, args.days
                )
                df = yield client.fetch_candles(instrument, gran, start, end)

                if df.empty:
                    logger.warning("  データなし: %s %s", instrument, gran)
                    continue

                logger.info("  取得完了: %d 本", len(df))
                out_path = Path(cfg.data.cache_dir) / f"{instrument}_{gran}.csv"
                save_candles_csv(df, out_path)
                logger.info("  保存先: %s", out_path)

            result["ok"] = True

        except Exception as exc:
            logger.error("取得エラー: %s", exc, exc_info=True)
            result["error"] = exc
        finally:
            client.disconnect()
            reactor.stop()

    reactor.callWhenRunning(run)
    reactor.run()

    if not result["ok"]:
        raise result["error"] or RuntimeError("不明なエラー")


# --------------------------------------------------------------------------
# エントリーポイント
# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="ローソク足データを取得してキャッシュする (cTrader / OANDA 対応)"
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--instrument", default=None,
        help="取引銘柄 (例: USD_JPY, XAU_USD)。省略時は config から取得"
    )
    parser.add_argument(
        "--granularity", default=None,
        help="足種 (例: M5, H1)。省略時は LTF と HTF の両方を取得"
    )
    parser.add_argument(
        "--days", type=int, default=365,
        help="取得する過去の日数 (デフォルト: 365)"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = get_logger("fxmaster.fetch", cfg.logging.level, cfg.logging.file)

    logger.info("ブローカー: %s (%s)", cfg.broker.name, cfg.broker.environment)

    if cfg.broker.name == "ctrader":
        _run_ctrader(args, cfg, logger)
    else:
        _run_oanda(args, cfg, logger)

    logger.info("データ取得が完了しました。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
