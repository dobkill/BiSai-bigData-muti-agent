from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import tempfile
from typing import Any


def stable_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def parse_date(value: str) -> str:
    parts = re.split(r"[.\-/]", value)
    return datetime(*(int(p) for p in parts)).date().isoformat()


def parse_time(value: Any) -> str | None:
    if isinstance(value, datetime):
        return value.isoformat(timespec="seconds")
    if value is None:
        return None
    for fmt in ("%Y.%m.%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(str(value), fmt).isoformat(timespec="seconds")
        except ValueError:
            pass
    return None


def write_json(path: Path, value: Any) -> None:
    write_text(path, json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # 原子替换，Ctrl+C 或进程崩溃不会留下半个缓存文件。
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False) as f:
        f.write(value)
        temp = Path(f.name)
    temp.replace(path)


def unique(values: list[Any]) -> list[Any]:
    result = []
    for value in values:
        if value is not None and value != "" and value not in result:
            result.append(value)
    return result

