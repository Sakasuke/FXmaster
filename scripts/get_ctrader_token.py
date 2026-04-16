"""cTrader Open API - OAuth2.0 アクセストークン取得スクリプト

【使い方】
1. config/config.yaml の broker セクションに client_id / client_secret を記載
   （または環境変数 CTRADER_CLIENT_ID / CTRADER_CLIENT_SECRET でも可）
2. このスクリプトを実行:
       python scripts/get_ctrader_token.py
3. ブラウザが開くので Axiory アカウントでログイン・認可
4. 成功すると .env ファイルに以下が保存される:
       CTRADER_ACCESS_TOKEN=...
       CTRADER_REFRESH_TOKEN=...
       CTRADER_TOKEN_EXPIRES_AT=...（UTC ISO形式）

【参考】cTrader Open API ドキュメント:
    https://connect.spotware.com/apps
"""
from __future__ import annotations

import argparse
import http.server
import json
import os
import sys
import threading
import urllib.parse
import webbrowser
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

# リポジトリルートをパスに追加
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fxmaster.utils.logger import get_logger

# --------------------------------------------------------------------------
# cTrader OAuth2 エンドポイント
# --------------------------------------------------------------------------
AUTH_BASE_URL = "https://connect.spotware.com/apps"
TOKEN_URL = f"{AUTH_BASE_URL}/token"
REDIRECT_URI = "http://localhost:8080"

logger = get_logger("fxmaster.get_token")


# --------------------------------------------------------------------------
# ローカルHTTPサーバー（認証コード受け取り用）
# --------------------------------------------------------------------------
class _CallbackHandler(http.server.BaseHTTPRequestHandler):
    """ブラウザからのリダイレクトを受け取り、認証コードを取り出す。"""

    captured_code: str | None = None
    captured_error: str | None = None

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)

        if "code" in params:
            _CallbackHandler.captured_code = params["code"][0]
            body = (
                "<html><body>"
                "<h2>✅ 認証成功！このタブを閉じて、ターミナルに戻ってください。</h2>"
                "</body></html>"
            ).encode("utf-8")
            self.send_response(200)
        else:
            _CallbackHandler.captured_error = params.get("error", ["unknown"])[0]
            body = (
                "<html><body>"
                "<h2>❌ 認証失敗。ターミナルのエラーを確認してください。</h2>"
                f"<p>{_CallbackHandler.captured_error}</p>"
                "</body></html>"
            ).encode("utf-8")
            self.send_response(400)

        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:  # type: ignore[override]
        # デフォルトのアクセスログを抑制
        pass


def _run_server(server: http.server.HTTPServer) -> None:
    """バックグラウンドスレッドでHTTPサーバーを起動。"""
    server.handle_request()  # 1リクエスト受け取ったら終了


# --------------------------------------------------------------------------
# トークン交換
# --------------------------------------------------------------------------
def _exchange_code(client_id: str, client_secret: str, code: str) -> dict:
    """認証コードをアクセストークンに交換する。"""
    resp = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": REDIRECT_URI,
            "client_id": client_id,
            "client_secret": client_secret,
        },
        timeout=30,
    )
    if not resp.ok:
        raise RuntimeError(
            f"トークン取得失敗: HTTP {resp.status_code}\n{resp.text[:500]}"
        )
    return resp.json()


def _refresh_token(client_id: str, client_secret: str, refresh_tok: str) -> dict:
    """リフレッシュトークンを使ってアクセストークンを更新する。"""
    resp = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_tok,
            "client_id": client_id,
            "client_secret": client_secret,
        },
        timeout=30,
    )
    if not resp.ok:
        raise RuntimeError(
            f"リフレッシュ失敗: HTTP {resp.status_code}\n{resp.text[:500]}"
        )
    return resp.json()


# --------------------------------------------------------------------------
# .env ファイルへの保存
# --------------------------------------------------------------------------
def _save_env(env_path: Path, tokens: dict) -> None:
    """取得したトークンを .env ファイルに書き込む。"""
    # expires_in は秒数で返ってくる（例: 2592000 = 30日）
    expires_in_sec: int = int(tokens.get("expires_in", 0))
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in_sec)

    lines: dict[str, str] = {}

    # 既存の .env を読み込んで他の設定を保持
    if env_path.exists():
        for line in env_path.read_text("utf-8").splitlines():
            if "=" in line and not line.startswith("#"):
                k, _, v = line.partition("=")
                lines[k.strip()] = v.strip()

    lines["CTRADER_ACCESS_TOKEN"] = tokens["access_token"]
    if "refresh_token" in tokens:
        lines["CTRADER_REFRESH_TOKEN"] = tokens["refresh_token"]
    lines["CTRADER_TOKEN_EXPIRES_AT"] = expires_at.isoformat()

    with env_path.open("w", encoding="utf-8") as f:
        f.write("# cTrader OAuth2 トークン（自動生成 - コミットしないこと）\n")
        for k, v in lines.items():
            f.write(f"{k}={v}\n")

    print(f"\n✅ トークンを保存しました: {env_path}")
    print(f"   Access Token : {tokens['access_token'][:20]}...")
    if "refresh_token" in tokens:
        print(f"   Refresh Token: {tokens['refresh_token'][:20]}...")
    print(f"   有効期限     : {expires_at.strftime('%Y-%m-%d %H:%M UTC')}")


# --------------------------------------------------------------------------
# メイン処理
# --------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="cTrader OAuth2 アクセストークンを取得して .env に保存します。"
    )
    parser.add_argument(
        "--client-id",
        default=os.environ.get("CTRADER_CLIENT_ID", ""),
        help="cTrader Open API Client ID（未指定なら環境変数 CTRADER_CLIENT_ID を使用）",
    )
    parser.add_argument(
        "--client-secret",
        default=os.environ.get("CTRADER_CLIENT_SECRET", ""),
        help="cTrader Open API Client Secret（未指定なら環境変数 CTRADER_CLIENT_SECRET を使用）",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="トークンを保存する .env ファイルのパス（デフォルト: .env）",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="既存のリフレッシュトークンでアクセストークンを更新する",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="ローカルコールバックサーバーのポート（デフォルト: 8080）",
    )
    args = parser.parse_args()

    # --- Client ID / Secret の確認 ---
    client_id = args.client_id
    client_secret = args.client_secret

    if not client_id:
        client_id = input("cTrader Client ID を入力してください: ").strip()
    if not client_secret:
        client_secret = input("cTrader Client Secret を入力してください: ").strip()

    if not client_id or not client_secret:
        print("❌ Client ID と Client Secret は必須です。")
        return 1

    env_path = Path(args.env_file)

    # --- リフレッシュモード ---
    if args.refresh:
        # .env から既存のリフレッシュトークンを読み込む
        existing: dict[str, str] = {}
        if env_path.exists():
            for line in env_path.read_text("utf-8").splitlines():
                if "=" in line and not line.startswith("#"):
                    k, _, v = line.partition("=")
                    existing[k.strip()] = v.strip()

        refresh_tok = existing.get("CTRADER_REFRESH_TOKEN", "")
        if not refresh_tok:
            print("❌ .env に CTRADER_REFRESH_TOKEN が見つかりません。")
            print("   先に通常モード（--refresh なし）でトークンを取得してください。")
            return 1

        print("♻️  リフレッシュトークンでアクセストークンを更新中...")
        tokens = _refresh_token(client_id, client_secret, refresh_tok)
        _save_env(env_path, tokens)
        return 0

    # --- 通常モード: 認証コードフロー ---
    redirect_uri = f"http://localhost:{args.port}"
    auth_url = (
        f"{AUTH_BASE_URL}/auth"
        f"?client_id={urllib.parse.quote(client_id)}"
        f"&redirect_uri={urllib.parse.quote(redirect_uri)}"
        f"&response_type=code"
        f"&scope=trading"
    )

    print("\n" + "=" * 60)
    print("🌐  ブラウザで Axiory の認証ページを開きます...")
    print("   ログイン後、自動的にこのスクリプトに戻ります。")
    print("=" * 60)
    print(f"\n認証URL（ブラウザが開かない場合は手動でアクセス）:\n{auth_url}\n")

    # ローカルサーバーをバックグラウンドで起動
    server_address = ("localhost", args.port)
    httpd = http.server.HTTPServer(server_address, _CallbackHandler)
    thread = threading.Thread(target=_run_server, args=(httpd,), daemon=True)
    thread.start()

    webbrowser.open(auth_url)
    print(f"⏳ ブラウザでの認証を待っています（localhost:{args.port} で待受中）...")

    thread.join(timeout=300)  # 5分でタイムアウト

    if _CallbackHandler.captured_error:
        print(f"❌ 認証エラー: {_CallbackHandler.captured_error}")
        return 1

    if not _CallbackHandler.captured_code:
        print("❌ タイムアウト: 5分以内に認証が完了しませんでした。")
        return 1

    code = _CallbackHandler.captured_code
    print(f"✅ 認証コードを取得しました: {code[:12]}...")
    print("🔄 アクセストークンに交換中...")

    tokens = _exchange_code(client_id, client_secret, code)
    _save_env(env_path, tokens)

    print("\n次のステップ:")
    print("  python scripts/fetch_data.py --instrument USDJPY --granularity M5 --days 90")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
