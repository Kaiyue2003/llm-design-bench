"""Load the batch launcher into an already prepared Colab session; never train.

Execute this commit-pinned file in the original notebook's namespace, after
Drive mounting. It leaves the frozen checkout and all existing results intact.
"""

from __future__ import annotations

import hashlib
import importlib.util
import re
import sys
import urllib.request
from pathlib import Path

BATCH_REVISION = "5f9bd5dc208d53f6f22684b76645168977ffd1b0"
BATCH_SHA256 = "459700324f4e71ebc5d5f1dc16a3a5fb4fcf910e0722fcfe1b5ce9efddeef0d9"
SUPPORT_SHA256 = "d607d5893fbf1b70b70d0d23762ba6394b727c620d564637a7463ed2f084b48e"
MAX_SCRIPT_BYTES = 1024 * 1024


def _fetch_launcher(cache_root: Path) -> Path:
    if not re.fullmatch(r"[0-9a-f]{40}", BATCH_REVISION):
        raise ValueError("batch launcher revision must be a full pinned commit")
    if not re.fullmatch(r"[0-9a-f]{64}", BATCH_SHA256):
        raise ValueError("batch launcher must have a pinned SHA256")
    destination = cache_root / BATCH_REVISION / "colab_batch.py"
    if any(path.is_symlink() for path in (cache_root, destination.parent, destination)):
        raise ValueError("launcher cache must not be a symlink")
    if destination.exists():
        with destination.open("rb") as handle:
            source = handle.read(MAX_SCRIPT_BYTES + 1)
    else:
        url = (
            "https://raw.githubusercontent.com/Kaiyue2003/llm-design-bench/"
            f"{BATCH_REVISION}/scripts/colab_batch.py"
        )
        with urllib.request.urlopen(url, timeout=30) as response:
            source = response.read(MAX_SCRIPT_BYTES + 1)
        if len(source) > MAX_SCRIPT_BYTES:
            raise ValueError("downloaded launcher exceeds its size limit")
        if hashlib.sha256(source).hexdigest() != BATCH_SHA256:
            raise ValueError("downloaded batch launcher hash mismatch; nothing loaded")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as handle:
            handle.write(source)
    if (
        len(source) > MAX_SCRIPT_BYTES
        or hashlib.sha256(source).hexdigest() != BATCH_SHA256
    ):
        raise ValueError("cached batch launcher hash mismatch; nothing loaded")
    return destination


def load_batch_runner(
    repo,
    assets,
    upstream,
    state,
    backups,
    *,
    neural_device="cuda",
    allow_environment_change=False,
):
    """Load checked orchestration code without changing experiment Python files."""
    repo = Path(repo).resolve()
    helper = repo / "scripts/colab_support.py"
    if (
        hashlib.sha256(helper.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        != SUPPORT_SHA256
    ):
        raise ValueError(
            "original single-job helper changed; do not mix launcher versions"
        )
    helpers = str(helper.parent)
    if helpers not in sys.path:
        sys.path.insert(0, helpers)
    import colab_support

    if Path(colab_support.__file__).resolve() != helper.resolve():
        raise ValueError("another colab_support module is already loaded")
    path = _fetch_launcher(repo.parent / "llmdm_batch_tools")
    name = "llmdm_colab_batch_" + BATCH_REVISION
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError("cannot load the verified batch launcher")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module.BatchRunner(
        assets=assets,
        data_recipes_root=upstream,
        state=state,
        backups=backups,
        neural_device=neural_device,
        allow_environment_change=allow_environment_change,
    )


if __name__ == "__main__":
    missing = [
        name
        for name in ("REPO", "ASSETS", "UPSTREAM", "STATE", "BACKUPS")
        if name not in globals()
    ]
    if missing:
        raise RuntimeError(
            "先完成原 notebook 的环境、数据校验和 Drive 挂载单元。缺少变量: "
            + ", ".join(missing)
        )
    batch = load_batch_runner(
        globals()["REPO"],
        globals()["ASSETS"],
        globals()["UPSTREAM"],
        globals()["STATE"],
        globals()["BACKUPS"],
        neural_device=globals().get("NEURAL_DEVICE", "cuda"),
        allow_environment_change=globals().get("ALLOW_ENVIRONMENT_CHANGE", False),
    )
    import pandas as pd
    from IPython.display import display

    queue = pd.DataFrame(batch.preview(phase="pilot"))
    display(
        queue[
            ["setting", "run_id", "seed", "device", "status"]
            + (["error"] if "error" in queue else [])
        ]
    )
    print(
        "批量入口加载成功；这里只预览，没有启动训练。complete 会校验后跳过，blocked 需要先检查。"
    )
