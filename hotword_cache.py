#!/usr/bin/env python3
"""voice-input: プロジェクト別ホットワードキャッシュ生成.

~/dev/ 配下のプロジェクトの README.md, CLAUDE.md を読み、
localmodel CLI で技術用語を抽出してキャッシュする。

使い方:
  python hotword_cache.py              # TTL有効（24h以内はスキップ）
  python hotword_cache.py --force      # 全プロジェクト再生成
  python hotword_cache.py --project X  # 特定プロジェクトのみ
  python hotword_cache.py --list       # キャッシュ内容を表示

キャッシュ: ~/.cache/voice-input/hotwords.json
夜間バッチ向け: crontab -e → 0 3 * * * /path/to/hotword_cache.py
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

CACHE_FILE = Path.home() / ".cache" / "voice-input" / "hotwords.json"
DEV_DIR = Path.home() / "dev"
TTL_HOURS = 24
MAX_CONTENT_CHARS = 4000
MAX_HOTWORDS = 50

SYSTEM_PROMPT = """\
あなたは音声認識(Whisper)の精度向上のためのキーワード抽出ツールです。
以下のプロジェクトドキュメントから、Whisperが誤認識しやすい技術用語を抽出してください。

抽出するもの:
- プロジェクト固有の技術用語（ライブラリ名、API名、設定名等）
- 固有名詞（ツール名、サービス名、プロトコル名等）
- プロジェクト固有のコマンド名・ファイル名
- 日本語カタカナ技術用語でWhisperが誤認識しやすいもの

除外するもの:
- 一般的すぎる語（API, LLM, VRAM, GPU, Python, JavaScript等）
- 一般的な英単語（server, client, config, test等）

出力形式:
- 1行に1語のみ
- 最大50語
- 説明・番号・記号は一切不要"""


def load_cache() -> dict:
    """キャッシュファイルを読み込み."""
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_cache(cache: dict):
    """キャッシュファイルに保存."""
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def extract_hotwords(project_name: str, content: str) -> list[str]:
    """localmodel CLIでホットワードを抽出."""
    if len(content) > MAX_CONTENT_CHARS:
        content = content[:MAX_CONTENT_CHARS] + "\n...(truncated)"

    messages = json.dumps({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"プロジェクト: {project_name}\n\n{content}"},
        ]
    })

    try:
        result = subprocess.run(
            [
                "localmodel",
                "-j", messages,
                "--think", "off",
                "--priority", "low",
                "--max-tokens", "512",
                "--num-ctx", "8192",
            ],
            capture_output=True, text=True, timeout=120,
        )
    except subprocess.TimeoutExpired:
        print(" timeout", file=sys.stderr)
        return []

    if result.returncode != 0:
        stderr = result.stderr.strip()[:200] if result.stderr else "unknown"
        print(f" error(rc={result.returncode}): {stderr}", file=sys.stderr)
        return []

    # Parse output: one word per line
    words = []
    for line in result.stdout.strip().split("\n"):
        word = line.strip()
        # Remove common list prefixes
        for prefix in ("- ", "・ ", "* ", "· "):
            if word.startswith(prefix):
                word = word[len(prefix):]
        # Remove numbering (1. 2. etc)
        if word and word[0].isdigit() and "." in word[:4]:
            word = word.split(".", 1)[1].strip()
        word = word.strip()
        if word and 1 < len(word) < 60:
            words.append(word)

    return words[:MAX_HOTWORDS]


def scan_projects(force: bool = False, target: str | None = None) -> dict:
    """~/dev/ 配下のプロジェクトをスキャンしてホットワードを抽出."""
    cache = load_cache()
    now = datetime.now()
    processed = 0
    skipped = 0

    if not DEV_DIR.is_dir():
        print(f"  ✗ {DEV_DIR} not found", file=sys.stderr)
        return cache

    for project_dir in sorted(DEV_DIR.iterdir()):
        if not project_dir.is_dir():
            continue

        name = project_dir.name
        if name.startswith("."):
            continue
        if target and name != target:
            continue

        # TTL check
        if not force and name in cache:
            try:
                updated = datetime.fromisoformat(cache[name]["updated_at"])
                age_hours = (now - updated).total_seconds() / 3600
                if age_hours < TTL_HOURS:
                    print(f"  ⏭ {name} (cached, {age_hours:.0f}h old)")
                    skipped += 1
                    continue
            except (KeyError, ValueError):
                pass

        # Read README.md + CLAUDE.md
        content_parts = []
        for fname in ("README.md", "CLAUDE.md"):
            fpath = project_dir / fname
            if fpath.exists():
                try:
                    text = fpath.read_text(encoding="utf-8")
                    content_parts.append(f"=== {fname} ===\n{text}")
                except Exception:
                    pass

        if not content_parts:
            continue

        content = "\n\n".join(content_parts)
        print(f"  ⟳ {name}...", end="", flush=True)

        t0 = time.time()
        words = extract_hotwords(name, content)
        elapsed = time.time() - t0

        cache[name] = {
            "path": str(project_dir),
            "updated_at": now.isoformat(),
            "hotwords": words,
        }

        print(f" {len(words)} words ({elapsed:.1f}s)")
        processed += 1

        # Save after each project (crash-safe)
        save_cache(cache)

    print(f"\nProcessed: {processed}, Skipped: {skipped}")
    return cache


def main():
    parser = argparse.ArgumentParser(
        description="voice-input: project hotword cache generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python hotword_cache.py              # Update stale caches (24h TTL)
  python hotword_cache.py --force      # Regenerate all
  python hotword_cache.py --project faster-whisper  # Single project
  python hotword_cache.py --list       # Show cached hotwords

Cron (nightly at 3am):
  0 3 * * * cd ~/dev/faster-whisper && .venv/bin/python hotword_cache.py
        """,
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Regenerate all (ignore TTL)",
    )
    parser.add_argument(
        "--project",
        help="Process only this project",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List cached hotwords",
    )
    args = parser.parse_args()

    if args.list:
        cache = load_cache()
        if not cache:
            print("Cache is empty.")
            return
        for name, entry in sorted(cache.items()):
            words = entry.get("hotwords", [])
            updated = entry.get("updated_at", "?")
            print(f"{name}: {len(words)} words (updated: {updated})")
            if words:
                print(f"  {', '.join(words[:15])}{'...' if len(words) > 15 else ''}")
        total = sum(len(e.get("hotwords", [])) for e in cache.values())
        print(f"\nTotal: {len(cache)} projects, {total} words")
        print(f"Cache: {CACHE_FILE}")
        return

    print(f"Scanning {DEV_DIR} for project hotwords...")
    print(f"Cache: {CACHE_FILE}")
    print(f"TTL: {TTL_HOURS}h {'(forced)' if args.force else ''}")
    print()

    cache = scan_projects(force=args.force, target=args.project)
    total = sum(len(e.get("hotwords", [])) for e in cache.values())
    print(f"\nTotal: {len(cache)} projects, {total} hotwords")
    print(f"Cache: {CACHE_FILE}")


if __name__ == "__main__":
    main()
