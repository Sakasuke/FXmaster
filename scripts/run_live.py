"""ライブ / デモ 自動売買ランナー.

cTrader と OANDA 両方に対応しています。
config.yaml の broker.name で自動的に切り替わります。

Usage:
    # cTrader (Axiory) デモ
    python scripts/run_live.py --instrument USD_JPY

    # OANDA デモ
    python scripts/run_live.py --instrument USD_JPY --env practice

    # シグナル判定のみ、注文なし (動作確認用)
    python scripts/run_live.py --instrument USD_JPY --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# --------------------------------------------------------------------------
# 共通ユーティリティ
# --------------------------------------------------------------------------

def _trading_hours_ok(now, start_str: str, end_str: str) -> bool:
    def _parse(hm: str):
        h, m = hm.split(":")
        return int(h), int(m)

    sh, sm = _parse(start_str)
    eh, em = _parse(end_str)
    start = now.replace(hour=sh, minute=sm, second=0, microsecond=0)
    end = now.replace(hour=eh, minute=em, second=0, microsecond=0)
    return start <= now <= end


# --------------------------------------------------------------------------
# cTrader パス（Twisted ネイティブ）
# --------------------------------------------------------------------------

def _run_ctrader(args, cfg) -> None:
    """cTrader 用のライブラン。Twisted LoopingCall で N 分ごとに戦略を評価する."""
    from datetime import datetime, timedelta, timezone

    from twisted.internet import defer, reactor, task

    from fxmaster.ai.model import TrendModel
    from fxmaster.config import load_config
    from fxmaster.data.ctrader_fetcher import (
        CTraderClient,
        CTraderCredentials,
        granularity_to_seconds,
    )
    from fxmaster.features.engineer import build_features
    from fxmaster.risk.manager import RiskManager
    from fxmaster.strategy import Direction, TrendScalpStrategy
    from fxmaster.utils.logger import get_logger
    from fxmaster.utils.pip_math import price_to_pips

    logger = get_logger("fxmaster.live", cfg.logging.level, cfg.logging.file)

    instrument = args.instrument or cfg.trading.instrument
    if args.env:
        cfg.broker.environment = args.env

    creds = CTraderCredentials(
        client_id=cfg.broker.client_id,
        client_secret=cfg.broker.client_secret,
        access_token=cfg.broker.access_token,
        account_id=int(cfg.broker.account_id),
        environment=cfg.broker.environment,
    )
    client = CTraderClient(creds)

    # 最新スポット価格 (bid, ask) — spot サブスクリプションで更新
    latest_bid = [0.0]
    latest_ask = [0.0]
    tick_running = [False]

    bar_seconds = granularity_to_seconds(cfg.trading.ltf_granularity)
    htf_seconds = granularity_to_seconds(cfg.trading.htf_granularity)
    lookback_bars = max(cfg.strategy.ema_slow * 3, 200)

    # ---------- スポット価格コールバック ----------
    def _on_message(c, message):
        msg_type = message.__class__.__name__
        if msg_type == "ProtoOASpotEvent":
            bid = message.bid / 100_000.0
            ask = message.ask / 100_000.0
            latest_bid[0] = bid
            latest_ask[0] = ask

    # ---------- 戦略評価ティック ----------
    @defer.inlineCallbacks
    def strategy_tick():
        if tick_running[0]:
            logger.debug("前回のティックがまだ実行中 — スキップ")
            return
        tick_running[0] = True

        try:
            now = datetime.now(timezone.utc)

            # 取引時間外チェック
            if not _trading_hours_ok(
                now,
                cfg.trading.trading_hours_utc.start,
                cfg.trading.trading_hours_utc.end,
            ):
                logger.debug("取引時間外 (%s UTC)", now.strftime("%H:%M"))
                return

            # ローソク足取得
            ltf_start = now - timedelta(seconds=bar_seconds * lookback_bars)
            htf_start = now - timedelta(seconds=htf_seconds * lookback_bars)

            ltf = yield client.fetch_candles(
                instrument, cfg.trading.ltf_granularity, ltf_start, now
            )
            htf = yield client.fetch_candles(
                instrument, cfg.trading.htf_granularity, htf_start, now
            )

            if len(ltf) < 50 or len(htf) < 50:
                logger.warning("足数不足 (ltf=%d htf=%d)、スキップ", len(ltf), len(htf))
                return

            # スプレッドチェック
            bid, ask = latest_bid[0], latest_ask[0]
            if bid > 0 and ask > 0:
                spread_pips = price_to_pips(ask - bid, instrument)
                if spread_pips > cfg.trading.max_spread_pips:
                    logger.info(
                        "スプレッド %.2f pips > 上限 %.2f pips — スキップ",
                        spread_pips, cfg.trading.max_spread_pips,
                    )
                    return
                price = ask  # エントリーは ask で計算
            else:
                price = float(ltf["close"].iloc[-1])

            # 特徴量・シグナル評価
            bundle = build_features(ltf, htf, instrument, cfg.strategy, None)
            row = bundle.features.iloc[-1]
            signal = strategy.evaluate_row(row, price)

            logger.info(
                "[%s] 方向=%s 確信度=%.2f 理由=%s",
                now.strftime("%H:%M UTC"),
                signal.direction.name,
                signal.confidence,
                signal.reason,
            )

            # ポジション照会
            open_pos = yield client.get_open_position(instrument)
            risk.state.open_positions = 1 if open_pos else 0

            can_open, reason = risk.can_open_new_position(now=now)

            if signal.is_actionable and open_pos is None and can_open:
                units = risk.position_size_units(
                    instrument=instrument,
                    entry_price=signal.entry_price,
                    stop_price=signal.stop_loss,
                )
                if units <= 0:
                    logger.info("ポジションサイズ 0 — スキップ")
                elif args.dry_run:
                    logger.info(
                        "[DRY-RUN] 発注予定: units=%d SL=%.5f TP=%.5f",
                        units if signal.direction == Direction.LONG else -units,
                        signal.stop_loss, signal.take_profit,
                    )
                else:
                    signed_units = (
                        units if signal.direction == Direction.LONG else -units
                    )
                    order = yield client.place_market_order(
                        instrument=instrument,
                        units=signed_units,
                        stop_loss=signal.stop_loss,
                        take_profit=signal.take_profit,
                    )
                    risk.register_open(now=now)
                    logger.info(
                        "ポジション開始: id=%s units=%d price=%.5f",
                        order.order_id, order.units, order.price,
                    )

            elif not can_open:
                logger.debug("リスクゲート: %s", reason)

        except Exception as exc:
            logger.error("ティックエラー: %s", exc, exc_info=True)
        finally:
            tick_running[0] = False

    # ---------- 初期化 ----------
    @defer.inlineCallbacks
    def setup():
        try:
            yield client.connect_and_auth()

            account = yield client.get_account_info()
            logger.info(
                "接続完了: account=%s balance=%.2f",
                account.account_id, account.balance,
            )

            # スポット価格サブスクリプション
            client.set_message_callback(_on_message)
            yield client.subscribe_spots(instrument)

            # AI モデルのロード
            try:
                _model = TrendModel.load(cfg.model.model_dir, name=f"{instrument}_trend")
                logger.info("AI モデル読み込み完了 (学習日: %s)", _model.trained_on)
                strategy._model = _model
            except FileNotFoundError:
                logger.warning("AI モデルなし — ルールベースで動作します")

            # LoopingCall でバーごとに戦略評価
            lc = task.LoopingCall(strategy_tick)
            lc.start(bar_seconds, now=True)

            logger.info(
                "自動売買 開始 (%s / %s 環境 / %d 秒足)",
                instrument, cfg.broker.environment, bar_seconds,
            )
            if args.dry_run:
                logger.info("★ DRY-RUN モード: 注文は発行されません ★")

        except Exception as exc:
            logger.error("初期化失敗: %s", exc, exc_info=True)
            reactor.stop()

    # モデル・リスクマネージャーの初期化（setup() の前に行う）
    try:
        _model_pre = TrendModel.load(cfg.model.model_dir, name=f"{instrument}_trend")
    except FileNotFoundError:
        _model_pre = None

    strategy = TrendScalpStrategy(cfg.strategy, model=_model_pre)
    risk = RiskManager(cfg.risk, initial_equity=1_000_000.0)  # setup() で更新

    reactor.callWhenRunning(setup)

    try:
        reactor.run()
    except KeyboardInterrupt:
        logger.info("Ctrl-C で停止しました")


# --------------------------------------------------------------------------
# OANDA パス（既存のポーリング処理）
# --------------------------------------------------------------------------

def _run_oanda(args, cfg) -> None:
    import time
    from datetime import datetime, timedelta, timezone

    from fxmaster.ai.model import TrendModel
    from fxmaster.broker import OandaBroker
    from fxmaster.data import OandaDataFetcher
    from fxmaster.data.fetcher import OandaCredentials, granularity_to_seconds
    from fxmaster.features.engineer import build_features
    from fxmaster.risk.manager import RiskManager
    from fxmaster.strategy import Direction, TrendScalpStrategy
    from fxmaster.utils.logger import get_logger
    from fxmaster.utils.pip_math import price_to_pips

    logger = get_logger("fxmaster.live", cfg.logging.level, cfg.logging.file)

    instrument = args.instrument or cfg.trading.instrument
    if args.env:
        cfg.broker.environment = args.env

    creds = OandaCredentials(
        account_id=cfg.broker.account_id,
        access_token=cfg.broker.access_token,
        environment=cfg.broker.environment,
    )
    fetcher = OandaDataFetcher(creds)
    broker = OandaBroker(creds)

    account = broker.account_info()
    logger.info(
        "OANDA %s 接続: account=%s balance=%.2f %s",
        cfg.broker.environment, account.account_id, account.balance, account.currency,
    )

    try:
        model = TrendModel.load(cfg.model.model_dir, name=f"{instrument}_trend")
        logger.info("AI モデル読み込み完了 (学習日: %s)", model.trained_on)
    except FileNotFoundError:
        model = None
        logger.warning("AI モデルなし — ルールベースで動作します")

    strategy = TrendScalpStrategy(cfg.strategy, model=model)
    risk = RiskManager(cfg.risk, initial_equity=account.equity)

    bar_seconds = granularity_to_seconds(cfg.trading.ltf_granularity)
    htf_seconds = granularity_to_seconds(cfg.trading.htf_granularity)
    lookback_bars = max(cfg.strategy.ema_slow * 3, 200)

    logger.info("自動売買 開始 (Ctrl-C で停止)")
    if args.dry_run:
        logger.info("★ DRY-RUN モード: 注文は発行されません ★")

    try:
        while True:
            now = datetime.now(timezone.utc)
            if not _trading_hours_ok(
                now, cfg.trading.trading_hours_utc.start, cfg.trading.trading_hours_utc.end
            ):
                logger.debug("取引時間外、スリープ")
                time.sleep(60)
                continue

            ltf_start = now - timedelta(seconds=bar_seconds * lookback_bars)
            htf_start = now - timedelta(seconds=htf_seconds * lookback_bars)
            ltf = fetcher.fetch_candles(instrument, cfg.trading.ltf_granularity, ltf_start, now)
            htf = fetcher.fetch_candles(instrument, cfg.trading.htf_granularity, htf_start, now)

            if len(ltf) < 50 or len(htf) < 50:
                logger.warning("足数不足 (ltf=%d htf=%d)", len(ltf), len(htf))
                time.sleep(bar_seconds)
                continue

            bundle = build_features(ltf, htf, instrument, cfg.strategy, None)
            row = bundle.features.iloc[-1]

            bid, ask = broker.get_price(instrument)
            spread_pips = price_to_pips(ask - bid, instrument)
            if spread_pips > cfg.trading.max_spread_pips:
                logger.info("スプレッド %.2f pips > 上限 — スキップ", spread_pips)
                time.sleep(bar_seconds)
                continue

            price = ask
            open_pos = broker.get_open_position(instrument)
            risk.state.open_positions = 1 if open_pos else 0

            signal = strategy.evaluate_row(row, price)
            logger.info(
                "[%s] 方向=%s 確信度=%.2f 理由=%s",
                now.strftime("%H:%M UTC"), signal.direction.name,
                signal.confidence, signal.reason,
            )

            can_open, reason = risk.can_open_new_position(now=now)

            if signal.is_actionable and open_pos is None and can_open:
                units = risk.position_size_units(
                    instrument=instrument,
                    entry_price=signal.entry_price,
                    stop_price=signal.stop_loss,
                )
                if units <= 0:
                    logger.info("ポジションサイズ 0 — スキップ")
                elif args.dry_run:
                    logger.info(
                        "[DRY-RUN] 発注予定: units=%d SL=%.5f TP=%.5f",
                        units if signal.direction == Direction.LONG else -units,
                        signal.stop_loss, signal.take_profit,
                    )
                else:
                    signed_units = (
                        units if signal.direction == Direction.LONG else -units
                    )
                    order = broker.place_market_order(
                        instrument=instrument,
                        units=signed_units,
                        stop_loss=signal.stop_loss,
                        take_profit=signal.take_profit,
                    )
                    risk.register_open(now=now)
                    logger.info(
                        "ポジション開始: id=%s units=%d price=%.5f",
                        order.order_id, order.units, order.price,
                    )
            elif not can_open:
                logger.debug("リスクゲート: %s", reason)

            time.sleep(bar_seconds)

    except KeyboardInterrupt:
        logger.info("Ctrl-C で停止しました")


# --------------------------------------------------------------------------
# エントリーポイント
# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="FXmaster 自動売買ランナー (cTrader / OANDA 対応)"
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--instrument", default=None,
        help="取引銘柄 (例: USD_JPY, XAU_USD)"
    )
    parser.add_argument(
        "--env", default=None,
        choices=["demo", "practice", "live"],
        help="ブローカー環境を上書き"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="シグナル評価のみ、注文なし"
    )
    args = parser.parse_args()

    from fxmaster.config import load_config
    cfg = load_config(args.config)

    if cfg.broker.name == "ctrader":
        _run_ctrader(args, cfg)
    else:
        _run_oanda(args, cfg)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
