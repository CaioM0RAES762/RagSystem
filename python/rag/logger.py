# rag/logger.py
import time


def fmt_secs(s: float) -> str:
    if s < 1:
        return f"{s*1000:.0f}ms"
    if s < 60:
        return f"{s:.1f}s"
    m = int(s // 60)
    r = s % 60
    return f"{m}m{r:.0f}s"


def log_step(title: str):
    print(f"\n🔷 {title}")


def log_info(msg: str):
    print(f"   • {msg}")


def log_ok(msg: str):
    print(f"   ✅ {msg}")


def log_warn(msg: str):
    print(f"   ⚠️  {msg}")


def log_err(msg: str):
    print(f"   ❌ {msg}")
