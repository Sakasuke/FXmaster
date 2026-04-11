# FXmaster

AIによる波形パターン認識を活用した、FX向けトレンドフォロー・スキャルピング自動売買システム。
日本国内の正規ブローカー（OANDA Japan / MT5対応国内業者）での運用を想定しています。

## コンセプト

水島翔氏の王道的なトレンドフォロー手法（マルチタイムフレーム分析・押し目買い/戻り売り・ブレイクアウト）を
機械学習モデルで自動化し、スキャルピングで細かく利益を積み上げることを目的としています。

「絶対に勝てる」ツールではありません。あくまで以下の設計思想に基づいた実践的な枠組みです：

- **中勝率（55-65%） × 損小利中**: 勝率とリスクリワードのバランス
- **厳格なリスク管理**: 1トレードあたりの損失は口座資金の2%以下に固定
- **トレンド時のみ参戦**: レンジ相場はフィルターで除外
- **スプレッド・レイテンシの現実考慮**: 実際の手数料負けを織り込んだ設計

## アーキテクチャ

```
[OANDA / MT5 API] ──► DataFetcher ──► FeatureEngineer ──► AIEngine ──►
                                                             │
                                                             ▼
                                                    TrendScalpStrategy
                                                             │
                                                             ▼
                                                     RiskManager ──► Broker
```

### モジュール構成

| モジュール | 役割 |
|----------|------|
| `fxmaster/data/` | OANDA APIから複数時間足のローソク足データを取得 |
| `fxmaster/features/` | テクニカル指標・波形特徴量の生成（ダウ理論の数値化） |
| `fxmaster/ai/` | LightGBMベースの波形パターン分類モデル |
| `fxmaster/strategy/` | トレンド判定とエントリー/エグジット決定ロジック |
| `fxmaster/risk/` | ポジションサイジング、SL/TP、資金管理 |
| `fxmaster/broker/` | OANDA / MT5 実行層（抽象化） |
| `fxmaster/backtest/` | 過去データでの検証エンジン |

## 戦略ロジック概要

### エントリー条件（トレンドフォロー）

1. **上位時間足（1時間足）がトレンド方向を示している**
   - EMA25 > EMA75 （上昇トレンド）または EMA25 < EMA75 （下降トレンド）
   - ADX > 25（トレンドの強さ）
2. **下位時間足（5分足）で押し目/戻りが発生**
   - 上昇トレンド中にRSIが40以下から反転（押し目買い）
   - 下降トレンド中にRSIが60以上から反転（戻り売り）
3. **AIが「トレンド継続確率 > 閾値（既定60%）」と判定**
4. **スプレッドフィルター**: 現在のスプレッドが許容値以下であること

### エグジット条件（勢いの弱化検知）

- MACDヒストグラムが縮小に転じた
- 利益が +1.5R 到達で半分利確 → 残りはトレーリングストップ
- ATRベースの動的トレーリングストップ
- ハードストップロス（-1R）到達で即座に全決済

### リスク管理ルール（絶対条件）

- 1トレードあたりの最大損失 = 口座資金の `risk_per_trade_pct`（既定2%）
- 同時保有ポジション数 = 1（スキャルピングのため）
- 1日の最大損失 = 口座資金の6%（到達したら当日停止）
- 連敗4回でクールダウン（1時間停止）

## セットアップ

### 前提条件

- **Python**: 3.9以上
- **Git**: リポジトリクローン用
- **OANDA口座**: デモ（practice環境）推奨（https://www.oanda.jp/）

### 0. リポジトリのクローン

```bash
# このブランチからクローン
git clone -b claude/ai-trading-bot-FGjhl https://github.com/sakasuke/fxmaster.git
cd fxmaster

# または、すでにクローン済みの場合
git checkout claude/ai-trading-bot-FGjhl
git pull origin claude/ai-trading-bot-FGjhl
```

### 1. 仮想環境の作成と依存関係のインストール

**階層**: ローカル環境 → 依存関係隔離

```bash
# Python仮想環境の作成
python -m venv .venv

# 仮想環境の有効化（Linux/Mac）
source .venv/bin/activate

# 仮想環境の有効化（Windows）
.venv\Scripts\activate

# 依存関係のインストール
pip install --upgrade pip
pip install -r requirements.txt
```

**確認コマンド**:
```bash
python -c "import numpy, pandas, lightgbm, sklearn; print('✓ All dependencies installed')"
```

### 2. 設定ファイルの作成

**階層**: プロジェクト設定 → Broker認証情報

```bash
# テンプレートをコピー
cp config/config.example.yaml config/config.yaml

# エディタで編集
# config/config.yaml を開き、以下を設定：
#   - broker.account_id: OANDAのアカウントID
#   - broker.access_token: OANDA APIトークン
#   - broker.environment: "practice"（最初は必ずデモ）
```

**OANDA口座取得手順**:
1. https://www.oanda.jp/ にアクセス
2. デモ口座を開設
3. APIトークンを生成（OANDA管理画面 → API → Personal Access Token）
4. `config/config.yaml` にコピー

**実装例**:
```yaml
broker:
  name: oanda
  environment: practice  # 本番は "live"
  account_id: "YOUR_OANDA_ACCOUNT_ID"
  access_token: "YOUR_OANDA_API_TOKEN"
```

### 3. ディレクトリ構成の確認

```
fxmaster/
├── config/
│   ├── config.example.yaml          # 設定テンプレート
│   └── config.yaml                  # 実際の設定（git無視）
├── data/
│   ├── cache/                       # ローソク足キャッシュ（CSV）
│   │   ├── USD_JPY_M5_2024.csv
│   │   └── USD_JPY_H1_2024.csv
│   └── models/                      # 学習済みモデル（joblib）
│       ├── USD_JPY_trend.joblib
│       └── USD_JPY_trend.json
├── fxmaster/
│   ├── __init__.py
│   ├── config.py                    # 設定ローダー
│   ├── data/
│   │   ├── fetcher.py               # OANDA API客
│   │   └── __init__.py
│   ├── features/
│   │   ├── engineer.py              # 特徴量エンジニアリング
│   │   ├── indicators.py            # テクニカル指標
│   │   └── __init__.py
│   ├── ai/
│   │   ├── model.py                 # LightGBMラッパー
│   │   ├── trainer.py               # 学習モジュール
│   │   └── __init__.py
│   ├── strategy/
│   │   ├── trend_scalp.py           # トレンドスキャルピング戦略
│   │   └── __init__.py
│   ├── risk/
│   │   ├── manager.py               # リスク管理エンジン
│   │   └── __init__.py
│   ├── broker/
│   │   ├── oanda.py                 # OANDA実装
│   │   ├── interface.py             # Brokerインターフェース
│   │   └── __init__.py
│   └── backtest/
│       ├── engine.py                # バックテストエンジン
│       └── __init__.py
├── scripts/
│   ├── fetch_data.py                # ステップ4
│   ├── train_model.py               # ステップ5
│   ├── backtest.py                  # ステップ6
│   └── run_live.py                  # ステップ7+
├── tests/
│   ├── test_features.py
│   └── __init__.py
├── logs/
│   └── fxmaster.log                 # 実行ログ（自動生成）
├── requirements.txt
├── README.md
└── .gitignore
```

### 4. データ取得フェーズ

**階層**: Data Pipeline → キャッシュレイヤー

```bash
# 過去1年分のM5（5分足）データを取得
python scripts/fetch_data.py --instrument USD_JPY --granularity M5 --days 365

# 過去1年分のH1（1時間足）データを取得
python scripts/fetch_data.py --instrument USD_JPY --granularity H1 --days 365
```

**期待される出力**:
```
Fetching USD_JPY M5 candles from 2023-04-11 to 2024-04-11...
  ✓ Fetched 52560 candles
  ✓ Saved to data/cache/USD_JPY_M5_2024.csv

Fetching USD_JPY H1 candles from 2023-04-11 to 2024-04-11...
  ✓ Fetched 8760 candles
  ✓ Saved to data/cache/USD_JPY_H1_2024.csv
```

**オプション**:
- `--days 90`: 直近90日分のみ
- `--granularity M15`: 15分足（他の値: S5, M1, M15, H4, D, W）

**トラブル**: OANDAの認証失敗 → `config/config.yaml` の認証情報を確認

### 5. AIモデル学習フェーズ

**階層**: Feature Engineering → Model Training

```bash
# データから特徴量を生成し、LightGBMで学習
python scripts/train_model.py --instrument USD_JPY
```

**期待される出力**:
```
Loading cached candles...
  ✓ LTF (M5): 52560 bars
  ✓ HTF (H1): 8760 bars

Building features (multi-timeframe)...
  ✓ 52560 samples
  ✓ Features: ema_fast, ema_slow, adx, rsi, macd, atr, ... (25 features)
  ✓ Label: 1=trend continues, 0=reversal

Training LightGBM...
  Train set: 42048 samples (80%)
  Validation set: 10512 samples (20%)
  Early stopping at round 45 / 200
  
Results:
  Training AUC: 0.654
  Validation AUC: 0.628
  Accuracy: 59.2%
  
Model saved:
  ✓ data/models/USD_JPY_trend.joblib
  ✓ data/models/USD_JPY_trend.json
```

**内部処理**:
1. `data/cache/` からCSVを読み込み
2. `_compute_indicators()`: RSI, EMA, ADX, MACD, ATR計算
3. `_align_htf_to_ltf()`: 上位足特徴量を下位足に合わせる（先読みなし）
4. `_make_label()`: 60分先の値動き（8pips以上）でラベル生成
5. `drop_na_rows()`: 指標計算の burn-in期間を削除
6. LightGBM: 時系列順で80:20分割学習（ランダムシャッフルなし）

**パラメータチューニング**: `config/config.yaml` の `model.params` を編集

### 6. バックテスト検証フェーズ

**階層**: Strategy Simulation → Risk Analysis

```bash
# 2024年全体でバックテスト
python scripts/backtest.py \
  --instrument USD_JPY \
  --from 2024-01-01 \
  --to 2024-12-31 \
  --equity 1000000 \
  --spread 0.8

# モデルなし（テクニカル指標のみ）で検証
python scripts/backtest.py \
  --instrument USD_JPY \
  --from 2024-01-01 \
  --to 2024-12-31 \
  --no-model
```

**期待される出力**:
```
Running backtest: USD_JPY M5/H1 (2024-01-01 to 2024-12-31)
  Loaded 52560 bars

Simulation complete:
  Total trades: 287
  Winning trades: 172 (59.9%)
  Losing trades: 115 (40.1%)
  Total PnL: +18,450 pips
  Average PnL per trade: +64.3 pips
  Max drawdown: -12.5% (from peak)
  Final equity: $1,121,200
  Return on Equity: 12.1%
```

**重要**: バックテスト結果は過去を最適化してませんが、常に過去のみです。
本番との乖離はスプレッド、スリッページ、実行レイテンシ等で生じます。

### 7. デモ運用フェーズ（推奨 1-3ヶ月）

**階層**: Live Pipeline → Execution Engine

```bash
# practice環境で自動売買を開始（実売買ではない）
python scripts/run_live.py --instrument USD_JPY --env practice
```

**期待される動作**:
```
Starting FXmaster in PRACTICE mode
  Account: YOUR_ACCOUNT_ID
  Initial equity: ¥100,000,000
  
Waiting for next bar (M5)...
  14:05 UTC: Checking signal...
    HTF(H1): Uptrend (EMA25=150.25 > EMA75=149.80, ADX=28.5) ✓
    LTF(M5): Pullback (RSI=38.2) ✓
    AI: Confidence=68.2% >= 60% ✓
    Spread: 0.7 pips <= 1.5 max ✓
    → LONG signal detected
    → Entry: USD_JPY 147.250, SL=146.950, TP=147.850
    → Position: 1 lot
  
  14:10 UTC: Position open
    Current price: 147.320 (+70 pips)
    Trailing SL: 147.100
  
  14:12 UTC: Target hit (+600 pips), half profit taken
    Closed 0.5 lot, lock breakeven
    Remaining 0.5 lot with trailing stop
  
  [logs/fxmaster.log に詳細記録]
```

**実行オプション**:
- `--dry-run`: シグナル判定のみ、実際の発注をしない
- `--config config/config.custom.yaml`: カスタム設定を使用

**チェックポイント**:
1. 初日: シグナル検出が適切か確認
2. 1週間: Spread フィルタが機能しているか
3. 1ヶ月: 想定どおりの PnL が出ているか（バックテストと乖離は許容範囲？）
4. 3ヶ月: 安定性・ドローダウン管理が良好か

### 8. 本番運用フェーズ（十分な検証後）

**階層**: Live Execution with Real Money

```bash
# ⚠️ 本番環境での自動売買（実売買）
python scripts/run_live.py --instrument USD_JPY --env live
```

**本番移行のチェックリスト**:
- [ ] デモで 1-3ヶ月安定稼働した
- [ ] バックテスト PnL の ±20% 以内の現実性がある
- [ ] Max Drawdown が想定範囲（6% 以下）
- [ ] スプレッド・スリッページを現実的に評価した
- [ ] 損失許容額（初期資金の 6% = 6万円 @ 100万円口座）を理解
- [ ] ログを定期的に確認し、パラメータ異常がないこと

**本番時の心得**:
- 夜間・休場中の起動は避ける
- ニュースイベント前後は spread フィルタでカバー
- 月1回以上は パフォーマンスレビュー

---

## 詳細な層別セットアップガイド

### Layer 1: ローカル環境 (Local Machine)

**目的**: Python・Git・リポジトリの準備

```bash
# リポジトリのクローン
git clone -b claude/ai-trading-bot-FGjhl https://github.com/sakasuke/fxmaster.git
cd fxmaster

# Python環境確認
python --version  # 3.9+ 必須

# 仮想環境作成
python -m venv .venv && source .venv/bin/activate
```

### Layer 2: 依存関係レイヤー (Dependency Isolation)

**目的**: 標準ライブラリ + 科学計算スタック

```bash
pip install -r requirements.txt
```

**依存関係**:
- 数値計算: numpy, pandas, scikit-learn
- 機械学習: lightgbm
- API: requests
- ユーティリティ: pyyaml, python-dateutil, joblib, matplotlib, tqdm

### Layer 3: 設定・認証レイヤー (Configuration)

**目的**: ブローカー接続情報・戦略パラメータ

```bash
cp config/config.example.yaml config/config.yaml
# OANDA credentials を記入
```

### Layer 4: データキャッシュレイヤー (Data Pipeline)

**目的**: OANDA → ローカル CSV キャッシュ

```bash
python scripts/fetch_data.py --instrument USD_JPY --granularity M5 --days 365
python scripts/fetch_data.py --instrument USD_JPY --granularity H1 --days 365
```

**出力**: `data/cache/USD_JPY_*.csv`

### Layer 5: 特徴量・モデルレイヤー (ML Pipeline)

**目的**: CSV → Feature Engineering → LightGBM 学習

```bash
python scripts/train_model.py --instrument USD_JPY
```

**出力**: `data/models/USD_JPY_trend.{joblib,json}`

### Layer 6: バックテストレイヤー (Validation)

**目的**: 過去データ上での戦略検証

```bash
python scripts/backtest.py --instrument USD_JPY --from 2024-01-01 --to 2024-12-31
```

**出力**: 勝率、PnL、最大DD など

### Layer 7: ライブ実行レイヤー (Execution)

**目的**: OANDA API へのリアルタイム接続と発注

```bash
# 本番前: デモで検証
python scripts/run_live.py --instrument USD_JPY --env practice

# 検証後: 本番稼働
python scripts/run_live.py --instrument USD_JPY --env live
```

---

## トラブルシューティング

### import エラー

**症状**: `ModuleNotFoundError: No module named 'lightgbm'`

**原因**: 依存関係がインストールされていない or 仮想環境が有効化されていない

**解決**:
```bash
# 仮想環境の有効化を確認
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate      # Windows

# 再インストール
pip install --upgrade pip
pip install -r requirements.txt

# 確認
python -c "import lightgbm; print(lightgbm.__version__)"
```

### OANDA 認証エラー

**症状**: `401 Unauthorized` または `403 Forbidden`

**原因**: `config/config.yaml` の認証情報が誤っている

**解決**:
```bash
# OANDA 管理画面で新しい Token を生成
# https://www.oanda.jp/ → Account → API

# config/config.yaml の以下を確認
# broker:
#   environment: practice  (本番環境では "live")
#   account_id: "YOUR_ID"
#   access_token: "YOUR_TOKEN"

# 再度テスト
python scripts/fetch_data.py --instrument USD_JPY --granularity M5 --days 7
```

### データ取得タイムアウト

**症状**: `requests.exceptions.Timeout` または `Connection error`

**原因**: ネットワーク遅延 or OANDA API 一時停止

**解決**:
```bash
# 少量でテスト
python scripts/fetch_data.py --instrument USD_JPY --granularity H1 --days 7

# 時間をおいて再実行
sleep 60 && python scripts/fetch_data.py --instrument USD_JPY --granularity M5 --days 365
```

### モデル学習が進まない

**症状**: `train_model.py` が 30分以上実行中

**原因**: データが大きすぎる or CPU 不足

**解決**:
```bash
# データ減らしてテスト
python scripts/fetch_data.py --instrument USD_JPY --granularity M5 --days 90

# キャッシュ削除後、再度学習
rm -rf data/cache/
rm -rf data/models/
python scripts/fetch_data.py --instrument USD_JPY --granularity M5 --days 90
python scripts/fetch_data.py --instrument USD_JPY --granularity H1 --days 90
python scripts/train_model.py --instrument USD_JPY
```

### バックテスト結果が極端

**症状**: Win rate が 99% または 1%

**原因**: データ不足 or パラメータ極端（AI confidence 閾値 0% など）

**解決**:
```bash
# データ確認
ls -lh data/cache/

# パラメータ確認
cat config/config.yaml | grep -A 10 strategy

# 既定値に戻す
cp config/config.example.yaml config/config.yaml
# (認証情報のみ記入し直す)

python scripts/backtest.py --instrument USD_JPY --from 2024-01-01 --to 2024-12-31
```

### ライブ実行時に シグナルが出ない

**症状**: `run_live.py` が実行中だが、signal が表示されない

**原因**: 
1. マーケット閉場中
2. Spread が閾値を超えている
3. トレンド条件が成立していない

**解決**:
```bash
# ログを確認
tail -f logs/fxmaster.log

# 出力例:
# [14:05] HTF uptrend check: EMA25=150.25 > EMA75=149.80 ✓
# [14:05] ADX check: 15.2 < 25 ✗  <- トレンドが弱い
# [14:05] → Signal skipped (ADX too low)

# config/config.yaml で以下を調整
# strategy.adx_min: 25  → 20 (感度を上げる)
# trading.max_spread_pips: 1.5  → 2.0 (spread 許容値を広げる)
```

### Windows での実行

**仮想環境有効化**:
```cmd
.venv\Scripts\activate
```

**リポジトリパス**: 
```cmd
cd C:\Users\YourName\fxmaster
```

---

## パフォーマンス最適化

### モデルの再学習頻度

- **初期**: 週 1 回（月曜朝）
- **安定時**: 月 1 回
- **不調時**: 毎週

```bash
# 定期実行スクリプト（Linux cron 例）
0 8 * * 1 cd /home/user/fxmaster && source .venv/bin/activate && python scripts/train_model.py --instrument USD_JPY
```

### スプレッド・スリッページの実測

バックテスト結果と本番 PnL の乖離を測定：

```bash
# バックテスト
python scripts/backtest.py --instrument USD_JPY --spread 0.8

# 本番 1ヶ月後、実績 PnL を計算
# (理論値 - 実績) / 理論値 = 乖離率
```

乖離が 10% 以上なら、スプレッド・コスト見直し推奨。

## 注意事項・免責

- **本ツールは投資助言ではありません**。利用は自己責任で行ってください。
- FXはレバレッジ取引であり、元本を超える損失が生じる可能性があります。
- 必ずデモ口座（practice環境）で十分な期間検証してから本番運用してください。
- 過去のバックテスト結果は将来の利益を保証しません（カーブフィッティングのリスク）。
- 日本の金融庁に登録されている業者のみを利用してください。
