"""
settings.py — .env 加载与配置校验（零依赖）。

为什么不用 python-dotenv：
  这个项目目前的第三方依赖只有 numpy。面试环境里 `pip install` 失败是
  真会发生的事，少一个依赖少一个风险点。下面 40 行覆盖了 .env 的全部
  常用语法。真想用 dotenv 也行，load_dotenv() 会自动让路（见文件末尾）。

优先级（后面的覆盖前面的）：
  .env 文件  <  真实环境变量  <  命令行参数
即已经 export 过的环境变量不会被 .env 悄悄改掉 —— 避免"我明明改了 .env
怎么没生效"这类现场浪费时间的问题。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

_LINE = re.compile(
    r"""^\s*
        (?:export\s+)?              # 允许 `export FOO=bar`
        ([A-Za-z_][A-Za-z0-9_]*)    # key
        \s*=\s*
        (.*?)                       # value
        \s*$""",
    re.VERBOSE,
)


def _unquote(v: str) -> str:
    if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
        inner = v[1:-1]
        # 双引号内解转义，单引号内保持字面（和 shell 一致）
        return inner.replace("\\n", "\n").replace("\\t", "\t") if v[0] == '"' else inner
    # 未加引号时，# 之前的部分才是值（行内注释）
    return v.split(" #", 1)[0].strip()


def load_dotenv(path: str = ".env", override: bool = False) -> Dict[str, str]:
    """把 .env 读进 os.environ。返回本次实际写入的键值。"""
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

        # embedding 可以来自另一家（Anthropic 自己不提供 embedding 模型）
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

    # ---------- 关键：告诉你现在能跑到第几级 ----------
    def max_level(self) -> int:
        if not self.chat_key:
            return 0            # 纯 BM25，无需任何 key
        if not self.embed_key:
            return 1            # 有 chat 无 embedding：编译 + 硬过滤 + BM25
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
            f"  -> 可运行到 --level {self.max_level()}"
            + ("" if self.can_submit() else "，但 EVAL_URL 未设置，只能 --no-submit"),
        ]
        for w in self.warnings():
            lines.append(f"  ! {w}")
        return "\n".join(lines)

    def warnings(self) -> List[str]:
        w: List[str] = []
        if self.chat_key and " " in self.chat_key.strip():
            w.append("chat_key 含空格，八成是 .env 里引号没配对")
        if self.provider == "gemini" and self.embed_model.startswith("gemini-embedding-2"):
            w.append(
                "gemini-embedding-2 对 list 输入会返回单个聚合向量，"
                "批量 embed 会静默错位；用 gemini-embedding-001 或逐条包 Content"
            )
        if self.embed_dim and self.embed_dim != 3072 and self.provider == "gemini":
            w.append(f"dim={self.embed_dim} 非预归一化，须自行 normalize（engine 已处理）")
        if self.eval_url.startswith("http://"):
            w.append("EVAL_URL 是明文 http，key 会裸奔")
        return w

    def require(self, level: int) -> None:
        """在真正烧 quota 之前 fail fast。"""
        if level > self.max_level():
            missing = "chat_key" if not self.chat_key else "embed_key"
            raise SystemExit(
                f"--level {level} 需要 {missing}，但 .env 里没读到。\n"
                f"当前最高可跑 --level {self.max_level()}。\n{self.report()}"
            )


# 如果装了 python-dotenv 且想用它，把这行取消注释即可覆盖上面的实现：
# from dotenv import load_dotenv  # noqa: F811

if __name__ == "__main__":
    print(Settings.load().report())
