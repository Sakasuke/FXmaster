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

### 1. 依存関係のインストール

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 設定ファイルの作成

```bash
cp config/config.example.yaml config/config.yaml
# config/config.yaml を編集して、OANDAのアカウントID・APIトークンを設定
```

OANDA Japan プロコース口座の取得方法は https://www.oanda.jp/ を参照してください。
最初は必ずデモ口座（practice環境）で動作確認すること。

### 3. データ取得と学習

```bash
# 過去データを取得してキャッシュ
python scripts/fetch_data.py --instrument USD_JPY --granularity M5 --days 365

# モデルを学習
python scripts/train_model.py --instrument USD_JPY
```

### 4. バックテスト

```bash
python scripts/backtest.py --instrument USD_JPY --from 2024-01-01 --to 2024-12-31
```

### 5. デモ運用（practice環境）

```bash
python scripts/run_live.py --instrument USD_JPY --env practice
```

### 6. 本番運用（十分な検証後のみ）

```bash
python scripts/run_live.py --instrument USD_JPY --env live
```

## 注意事項・免責

- **本ツールは投資助言ではありません**。利用は自己責任で行ってください。
- FXはレバレッジ取引であり、元本を超える損失が生じる可能性があります。
- 必ずデモ口座（practice環境）で十分な期間検証してから本番運用してください。
- 過去のバックテスト結果は将来の利益を保証しません（カーブフィッティングのリスク）。
- 日本の金融庁に登録されている業者のみを利用してください。
