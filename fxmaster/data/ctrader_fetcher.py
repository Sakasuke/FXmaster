"""cTrader Open API クライアント (Twisted ベース).

このモジュールはすべて Twisted の Deferred を返します。
fetch_data.py / run_live.py は reactor.run() の中で呼び出してください。

価格エンコーディングについて:
  cTrader の ProtoOATrendbar は価格を整数で保持します。
    actual_low   = bar.low / 100_000
    actual_open  = (bar.low + bar.deltaOpen)  / 100_000
    actual_close = (bar.low + bar.deltaClose) / 100_000
    actual_high  = (bar.low + bar.deltaHigh)  / 100_000

volume 単位の変換:
  cTrader API の volume: 100 = 1 ロット = 100,000 通貨単位
  RiskManager.position_size_units() は通貨単位を返すため、
    ctrader_volume = max(1, risk_units // 1000)
  で変換します（1ロット = 100 * 1000 = 100,000 通貨単位）。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from twisted.internet import defer

logger = logging.getLogger("fxmaster.data.ctrader_fetcher")

# --------------------------------------------------------------------------
# エンドポイント定数
# --------------------------------------------------------------------------
DEMO_HOST = "demo.ctraderapi.com"
LIVE_HOST = "live.ctraderapi.com"
API_PORT = 5035

# 足種文字列 → ProtoOATrendbarPeriod 整数値
_PERIOD_MAP: dict[str, int] = {
    "M1": 1, "M2": 2, "M3": 3, "M4": 4, "M5": 5,
    "M10": 6, "M15": 7, "M30": 8,
    "H1": 9, "H4": 10, "H12": 11,
    "D": 12, "W": 13,
}

# 足種 → 秒数
_PERIOD_SECONDS: dict[str, int] = {
    "M1": 60, "M2": 120, "M3": 180, "M4": 240, "M5": 300,
    "M10": 600, "M15": 900, "M30": 1800,
    "H1": 3600, "H4": 14400, "H12": 43200,
    "D": 86400, "W": 604800,
}

_MAX_BARS = 5000  # 1 リクエストあたりの上限


# --------------------------------------------------------------------------
# 認証情報
# --------------------------------------------------------------------------

@dataclass
class CTraderCredentials:
    """cTrader Open API への接続に必要な認証情報."""

    client_id: str
    client_secret: str
    access_token: str
    account_id: int       # 数値の ctidTraderAccountId
    environment: str = "demo"  # "demo" または "live"

    @property
    def host(self) -> str:
        return DEMO_HOST if self.environment == "demo" else LIVE_HOST


# --------------------------------------------------------------------------
# ユーティリティ
# --------------------------------------------------------------------------

def instrument_to_ctrader(instrument: str) -> str:
    """OANDA 形式 (USD_JPY) を cTrader 形式 (USDJPY) に変換する."""
    return instrument.replace("_", "")


def granularity_to_seconds(granularity: str) -> int:
    """足種文字列を秒数に変換する."""
    if granularity not in _PERIOD_SECONDS:
        raise ValueError(f"未対応の granularity: {granularity}。対応: {sorted(_PERIOD_SECONDS)}")
    return _PERIOD_SECONDS[granularity]


def _decode_trendbar(bar) -> dict:
    """ProtoOATrendbar を OHLCV の dict に変換する.

    cTrader はすべての価格を整数 (1/100_000 単位) で保持する。
    deltaOpen / deltaClose / deltaHigh は low からの差分。
    """
    low = bar.low / 100_000.0
    open_ = (bar.low + bar.deltaOpen) / 100_000.0
    close = (bar.low + bar.deltaClose) / 100_000.0
    high = (bar.low + bar.deltaHigh) / 100_000.0
    ts = datetime.fromtimestamp(bar.utcTimestampInMinutes * 60, tz=timezone.utc)
    return {
        "time": ts,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": int(bar.volume),
    }


# --------------------------------------------------------------------------
# メインクライアント
# --------------------------------------------------------------------------

class CTraderClient:
    """cTrader Open API クライアント。

    すべてのメソッドは Twisted の Deferred を返します。
    Twisted reactor が実行中の状態で呼び出してください。

    典型的な使い方 (fetch_data.py 等):
        @defer.inlineCallbacks
        def setup():
            client = CTraderClient(credentials)
            yield client.connect_and_auth()
            df = yield client.fetch_candles("USD_JPY", "M5", start, end)
            client.disconnect()
            reactor.stop()

        reactor.callWhenRunning(setup)
        reactor.run()
    """

    def __init__(self, credentials: CTraderCredentials) -> None:
        self.credentials = credentials
        self._client = None
        self._symbol_cache: dict[str, int] = {}   # cTrader名 → symbolId
        self._position_ids: dict[str, int] = {}   # instrument → positionId

    # ------------------------------------------------------------------
    # 接続・認証
    # ------------------------------------------------------------------

    @defer.inlineCallbacks
    def connect_and_auth(self):
        """TCP 接続 → アプリ認証 → アカウント認証 を順番に実行する."""
        from ctrader_open_api import Client, TcpProtocol
        from ctrader_open_api.messages.OpenApiMessages_pb2 import (
            ProtoOAApplicationAuthReq,
            ProtoOAAccountAuthReq,
        )

        self._client = Client(self.credentials.host, API_PORT, TcpProtocol)

        # 接続完了を待つ Deferred
        connected_d: defer.Deferred = defer.Deferred()

        def _on_connected(client, _):
            if not connected_d.called:
                connected_d.callback(None)

        self._client.setConnectedCallback(_on_connected)
        self._client.startService()
        yield connected_d
        logger.info("cTrader に接続しました (%s)", self.credentials.environment)

        # アプリケーション認証
        app_req = ProtoOAApplicationAuthReq()
        app_req.clientId = self.credentials.client_id
        app_req.clientSecret = self.credentials.client_secret
        yield self._client.send(app_req)
        logger.info("アプリケーション認証 完了")

        # アカウント認証
        acc_req = ProtoOAAccountAuthReq()
        acc_req.ctidTraderAccountId = self.credentials.account_id
        acc_req.accessToken = self.credentials.access_token
        yield self._client.send(acc_req)
        logger.info("アカウント %d 認証 完了", self.credentials.account_id)

    def set_message_callback(self, callback) -> None:
        """全受信メッセージ (SpotEvent, ExecutionEvent 等) のコールバックを設定する.

        callback(client, message) の形式で呼ばれる。
        """
        if self._client is not None:
            self._client.setMessageReceivedCallback(callback)

    def disconnect(self) -> None:
        """接続を切断する."""
        if self._client is not None:
            self._client.stopService()
            logger.info("cTrader から切断しました")

    # ------------------------------------------------------------------
    # シンボル ID 解決
    # ------------------------------------------------------------------

    @defer.inlineCallbacks
    def get_symbol_id(self, instrument: str) -> int:
        """シンボル名 (例: "USD_JPY") から cTrader の symbolId を返す.

        初回はサーバーへ問い合わせ、以降はキャッシュを利用する。
        """
        from ctrader_open_api.messages.OpenApiMessages_pb2 import ProtoOASymbolsListReq

        name = instrument_to_ctrader(instrument)
        if name in self._symbol_cache:
            return self._symbol_cache[name]

        req = ProtoOASymbolsListReq()
        req.ctidTraderAccountId = self.credentials.account_id
        res = yield self._client.send(req)

        for sym in res.symbol:
            self._symbol_cache[sym.symbolName] = sym.symbolId

        if name not in self._symbol_cache:
            known = sorted(self._symbol_cache.keys())[:30]
            raise ValueError(
                f"シンボル '{name}' が見つかりません。\n"
                f"利用可能なシンボル（先頭30件）: {known}\n"
                f"config.yaml の instrument を上記のいずれかに変更してください。"
            )
        return self._symbol_cache[name]

    # ------------------------------------------------------------------
    # ローソク足取得
    # ------------------------------------------------------------------

    @defer.inlineCallbacks
    def fetch_candles(
        self,
        instrument: str,
        granularity: str,
        start: datetime,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """指定期間のローソク足データを DataFrame で返す.

        Args:
            instrument: "USD_JPY" や "XAU_USD" など
            granularity: "M1","M5","M15","H1","H4","D" など
            start: 取得開始日時 (UTC)
            end:   取得終了日時 (UTC)。省略時は現在時刻

        Returns:
            columns: time, open, high, low, close, volume
        """
        from ctrader_open_api.messages.OpenApiMessages_pb2 import ProtoOAGetTrendbarsReq

        if end is None:
            end = datetime.now(timezone.utc)
        if granularity not in _PERIOD_MAP:
            raise ValueError(
                f"未対応の granularity: '{granularity}'。"
                f"対応: {sorted(_PERIOD_MAP.keys())}"
            )

        symbol_id = yield self.get_symbol_id(instrument)
        period_int = _PERIOD_MAP[granularity]
        bar_ms = _PERIOD_SECONDS[granularity] * 1000
        from_ms = int(start.timestamp() * 1000)
        to_ms = int(end.timestamp() * 1000)

        all_bars = []
        cursor_ms = from_ms

        while cursor_ms < to_ms:
            req = ProtoOAGetTrendbarsReq()
            req.ctidTraderAccountId = self.credentials.account_id
            req.symbolId = symbol_id
            req.period = period_int
            req.fromTimestamp = cursor_ms
            req.toTimestamp = to_ms
            req.count = _MAX_BARS

            res = yield self._client.send(req)
            bars = list(res.trendbar)
            if not bars:
                break

            all_bars.extend(bars)
            last_ts_ms = bars[-1].utcTimestampInMinutes * 60 * 1000
            cursor_ms = last_ts_ms + bar_ms

            logger.debug(
                "取得済み %d 本 / %s %s (最終: %s)",
                len(all_bars), instrument, granularity,
                datetime.fromtimestamp(last_ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M"),
            )

        if not all_bars:
            logger.warning("%s %s: データなし (期間: %s〜%s)", instrument, granularity, start, end)
            return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

        rows = [_decode_trendbar(b) for b in all_bars]
        df = (
            pd.DataFrame(rows)
            .drop_duplicates(subset="time")
            .sort_values("time")
            .reset_index(drop=True)
        )
        logger.info(
            "%s %s: %d 本取得", instrument, granularity, len(df)
        )
        return df

    # ------------------------------------------------------------------
    # リアルタイム価格
    # ------------------------------------------------------------------

    @defer.inlineCallbacks
    def subscribe_spots(self, instrument: str) -> None:
        """指定シンボルのスポット価格を購読開始する.

        価格更新は set_message_callback() で登録したコールバックで受け取る。
        ProtoOASpotEvent として来るので message.__class__.__name__ で判定できる。
        """
        from ctrader_open_api.messages.OpenApiMessages_pb2 import ProtoOASubscribeSpotsReq

        symbol_id = yield self.get_symbol_id(instrument)
        req = ProtoOASubscribeSpotsReq()
        req.ctidTraderAccountId = self.credentials.account_id
        req.symbolId.append(symbol_id)
        yield self._client.send(req)
        logger.info(
            "スポット価格購読 開始: %s (symbolId=%d)", instrument, symbol_id
        )

    # ------------------------------------------------------------------
    # アカウント情報
    # ------------------------------------------------------------------

    @defer.inlineCallbacks
    def get_account_info(self):
        """口座残高・資産情報を AccountInfo として返す."""
        from ctrader_open_api.messages.OpenApiMessages_pb2 import ProtoOATraderReq
        from fxmaster.broker.base import AccountInfo

        req = ProtoOATraderReq()
        req.ctidTraderAccountId = self.credentials.account_id
        res = yield self._client.send(req)
        trader = res.trader
        # cTrader は残高を 1/100 で保持 (例: 100000 = 1000.00 USD)
        balance = trader.balance / 100.0

        return AccountInfo(
            account_id=str(self.credentials.account_id),
            currency=str(getattr(trader, "depositAssetId", "USD")),
            balance=balance,
            equity=balance,
        )

    # ------------------------------------------------------------------
    # ポジション照会
    # ------------------------------------------------------------------

    @defer.inlineCallbacks
    def get_open_position(self, instrument: str):
        """指定シンボルのオープンポジションを返す。なければ None。

        positionId を内部キャッシュに保存するので close_position() と連携できる。
        """
        from ctrader_open_api.messages.OpenApiMessages_pb2 import ProtoOAReconcileReq
        from fxmaster.broker.base import OpenPosition

        symbol_id = yield self.get_symbol_id(instrument)
        req = ProtoOAReconcileReq()
        req.ctidTraderAccountId = self.credentials.account_id
        res = yield self._client.send(req)

        for pos in res.position:
            if pos.tradeData.symbolId == symbol_id:
                pos_id = pos.positionId
                self._position_ids[instrument] = pos_id
                upnl = getattr(pos, "unrealizedPnl", 0) / 100.0
                return OpenPosition(
                    instrument=instrument,
                    units=pos.tradeData.volume,
                    average_price=pos.price / 100_000.0,
                    unrealized_pnl=upnl,
                    position_id=pos_id,
                )
        return None

    # ------------------------------------------------------------------
    # 注文
    # ------------------------------------------------------------------

    @defer.inlineCallbacks
    def place_market_order(
        self,
        instrument: str,
        units: int,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ):
        """成行注文を発注する。

        Args:
            units: 正値 = BUY, 負値 = SELL
                   RiskManager の position_size_units() の戻り値を渡す想定
                   (通貨単位)。内部で cTrader volume = units // 1000 に変換。

        Returns:
            OrderResult
        """
        from ctrader_open_api.messages.OpenApiMessages_pb2 import ProtoOANewOrderReq
        from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import (
            ProtoOAOrderType,
            ProtoOATradeSide,
        )
        from fxmaster.broker.base import OrderResult

        symbol_id = yield self.get_symbol_id(instrument)

        # 通貨単位 → cTrader volume (100 = 1ロット = 100,000通貨)
        volume = max(1, abs(units) // 1000)
        side = ProtoOATradeSide.BUY if units > 0 else ProtoOATradeSide.SELL

        req = ProtoOANewOrderReq()
        req.ctidTraderAccountId = self.credentials.account_id
        req.symbolId = symbol_id
        req.orderType = ProtoOAOrderType.MARKET
        req.tradeSide = side
        req.volume = volume
        if stop_loss is not None:
            req.stopLoss = int(round(stop_loss * 100_000))
        if take_profit is not None:
            req.takeProfit = int(round(take_profit * 100_000))

        logger.info(
            "成行注文: %s %s vol=%d (SL=%s TP=%s)",
            "BUY" if units > 0 else "SELL",
            instrument, volume,
            f"{stop_loss:.5f}" if stop_loss else "なし",
            f"{take_profit:.5f}" if take_profit else "なし",
        )

        # send() は ProtoOAExecutionEvent を clientMsgId で照合して返す
        res = yield self._client.send(req)

        # 約定価格とオーダーIDを取り出す
        exec_price = 0.0
        order_id = ""
        if hasattr(res, "deal") and res.deal:
            exec_price = res.deal.executionPrice / 100_000.0
            order_id = str(res.deal.dealId)
        elif hasattr(res, "order") and res.order:
            exec_price = getattr(res.order, "executionPrice", 0) / 100_000.0
            order_id = str(res.order.orderId)

        return OrderResult(
            order_id=order_id,
            instrument=instrument,
            units=units,
            price=exec_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            raw={},
        )

    # ------------------------------------------------------------------
    # ポジション決済
    # ------------------------------------------------------------------

    @defer.inlineCallbacks
    def close_position(self, instrument: str) -> None:
        """指定シンボルのオープンポジションを全決済する."""
        from ctrader_open_api.messages.OpenApiMessages_pb2 import ProtoOAClosePositionReq

        position = yield self.get_open_position(instrument)
        if position is None:
            logger.info("%s: オープンポジションなし。決済スキップ", instrument)
            return

        pos_id = self._position_ids.get(instrument, 0)
        if pos_id == 0:
            logger.error("%s: positionId が不明。決済できません", instrument)
            return

        req = ProtoOAClosePositionReq()
        req.ctidTraderAccountId = self.credentials.account_id
        req.positionId = pos_id
        req.volume = abs(position.units)
        yield self._client.send(req)
        logger.info(
            "%s: ポジション決済完了 (positionId=%d, volume=%d)",
            instrument, pos_id, abs(position.units),
        )
