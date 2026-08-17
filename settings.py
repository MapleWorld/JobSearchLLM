"""
settings.py — .env loading and config validation (zero dependencies).

Why not python-dotenv:
  numpy is this project's only third-party dependency. `pip install`
  failing in an interview environment is a real event, and every extra
  dependency is another way to lose time. The ~40 lines below cover all
  the common .env syntax. If you would rather use python-dotenv, the
  import at the end of this file lets it take over.

Precedence (later wins):
  .env file  <  real environment variables  <  CLI flags
So an already-exported variable is never silently overwritten by .env,
which avoids the classic "I edited .env, why did nothing change?" time
sink during a live session.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

_LINE = re.compile(
    r"""^\s*
        (?:export\s+)?              # allow `export FOO=bar`
        ([A-Za-z_][A-Za-z0-9_]*)    # key
        \s*=\s*
        (.*?)                       # value
        \s*$""",
    re.VERBOSE,
)


def _unquote(v: str) -> str:
    if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
        inner = v[1:-1]
        # Double quotes expand escapes; single quotes stay literal (shell-like)
        return inner.replace("\\n", "\n").replace("\\t", "\t") if v[0] == '"' else inner
    # When unquoted, only the text before " #" is the value (inline comment)
    return v.split(" #", 1)[0].strip()


def load_dotenv(path: str = ".env", override: bool = False) -> Dict[str, str]:
    """Load .env into os.environ. Returns only the keys actually written."""
    loaded: Dict[str, str] = {}
    if not os.path.exists(path):
        return loaded
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            m = _LINE.match(line)
            if not m:
                continue
            k, v = m.group(1), _unquote(m.group(2))
            if override or k not in os.environ:
                os.environ[k] = v
                loaded[k] = v
    return loaded


# ======================================================================

@dataclass
class Settings:
    provider: str = "gemini"          # gemini | anthropic | openai
    chat_key: str = ""
    embed_key: str = ""
    chat_model: str = ""
    embed_model: str = ""
    embed_dim: Optional[int] = None
    eval_url: str = ""
    eval_key: str = ""

    @classmethod
    def load(cls, dotenv_path: str = ".env") -> "Settings":
        load_dotenv(dotenv_path)
        g = os.environ.get
        provider = (g("LLM_PROVIDER") or "gemini").lower()

        chat_key = {
            "gemini": g("GEMINI_API_KEY") or g("GOOGLE_API_KEY") or "",
            "anthropic": g("ANTHROPIC_API_KEY") or "",
            "openai": g("OPENAI_API_KEY") or "",
        }.get(provider, "")

        # Embeddings may come from a different vendor (Anthropic ships no embedding model)
        embed_key = (
            g("EMBED_API_KEY")
            or {
                "gemini": g("GEMINI_API_KEY") or g("GOOGLE_API_KEY") or "",
                "anthropic": g("VOYAGE_API_KEY") or g("OPENAI_API_KEY") or "",
                "openai": g("OPENAI_API_KEY") or "",
            }.get(provider, "")
        )

        defaults = {
            "gemini": ("gemini-3.7-flash", "gemini-embedding-001", 768),
            "anthropic": ("claude-sonnet-4-6", "voyage-3.5", None),
            "openai": ("gpt-4.1-mini", "text-embedding-3-small", None),
        }.get(provider, ("", "", None))

        dim = g("EMBED_DIM")
        return cls(
            provider=provider,
            chat_key=chat_key,
            embed_key=embed_key,
            chat_model=g("CHAT_MODEL") or defaults[0],
            embed_model=g("EMBED_MODEL") or defaults[1],
            embed_dim=int(dim) if dim else defaults[2],
            eval_url=g("EVAL_URL") or "",
            eval_key=g("EVAL_KEY") or "",
        )

    # ---------- Key: report which pipeline level is currently reachable ----------
    def max_level(self) -> int:
        if not self.chat_key:
            return 0            # Pure BM25, needs no API key at all
        if not self.embed_key:
            return 1            # Chat but no embeddings: compile + hard filter + BM25
        return 3

    def can_submit(self) -> bool:
        return bool(self.eval_url)

    def report(self) -> str:
        def mask(s: str) -> str:
            return f"{s[:6]}…{s[-4:]} ({len(s)} chars)" if len(s) > 12 else ("SET" if s else "—")

        lines = [
            "[SETTINGS]",
            f"  provider    : {self.provider}",
            f"  chat_model  : {self.chat_model}",
            f"  chat_key    : {mask(self.chat_key)}",
            f"  embed_model : {self.embed_model}"
            + (f" (dim={self.embed_dim})" if self.embed_dim else ""),
            f"  embed_key   : {mask(self.embed_key)}",
            f"  eval_url    : {self.eval_url or '—'}",
            f"  eval_key    : {mask(self.eval_key)}",
            f"  -> can run up to --level {self.max_level()}"
            + ("" if self.can_submit() else ", but EVAL_URL is unset, so --no-submit only"),
        ]
        for w in self.warnings():
            lines.append(f"  ! {w}")
        return "\n".join(lines)

    def warnings(self) -> List[str]:
        w: List[str] = []
        if self.chat_key and " " in self.chat_key.strip():
            w.append("chat_key contains a space - most likely unbalanced quotes in .env")
        if self.provider == "gemini" and self.embed_model.startswith("gemini-embedding-2"):
            w.append(
                "gemini-embedding-2 returns ONE aggregated vector for a list input, "
                "so batch embedding silently misaligns. Use gemini-embedding-001, "
                "or wrap each input in its own Content object."
            )
        if self.embed_dim and self.embed_dim != 3072 and self.provider == "gemini":
            w.append(f"dim={self.embed_dim} is not pre-normalized; vectors must be "
                     f"normalized manually (engine.retrieve already does this)")
        if self.eval_url.startswith("http://"):
            w.append("EVAL_URL uses plaintext http - the key travels unencrypted")
        return w

    def require(self, level: int) -> None:
        """Fail fast before burning any quota."""
        if level > self.max_level():
            missing = "chat_key" if not self.chat_key else "embed_key"
            raise SystemExit(
                f"--level {level} requires {missing}, but it was not found in .env.\n"
                f"Highest runnable level right now is --level {self.max_level()}.\n{self.report()}"
            )


# If python-dotenv is installed and you prefer it, uncomment to override the above:
# from dotenv import load_dotenv  # noqa: F811

if __name__ == "__main__":
    print(Settings.load().report())
