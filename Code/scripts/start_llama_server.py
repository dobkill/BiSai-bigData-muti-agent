"""Start a local CUDA llama-server with explicit batching and per-slot context."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.parse import urlsplit

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "src"))

from fusion_agent.config import load_settings


def server_command(settings, executable=None, model_path=None, slots=None):
    endpoint = settings.llm.llama_cpp
    address = urlsplit(endpoint.base_url)
    if address.scheme != "http" or address.hostname not in {"localhost", "127.0.0.1", "::1"} or address.path.rstrip("/") != "/v1":
        raise ValueError("启动脚本需要本机 http://localhost:端口/v1 地址")
    path = Path(model_path).resolve() if model_path else endpoint.model_path
    if path is None:
        raise ValueError("请在 llm.llama_cpp.model_path 或 --model-path 填写 GGUF 文件路径")
    count = slots if slots is not None else endpoint.server_slots
    if not 1 <= count <= 32:
        raise ValueError("--slots 必须在 1–32 之间")
    return [executable or endpoint.executable, "--model", str(path), "--alias", endpoint.model,
            "--host", address.hostname, "--port", str(address.port or 8080),
            "--n-gpu-layers", "99", "--parallel", str(count),
            "--ctx-size", str(endpoint.num_ctx * count), "--cont-batching", "--jinja",
            "--no-context-shift"]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CODE / "config.toml")
    parser.add_argument("--executable", help="CUDA 版 llama-server 路径，覆盖配置")
    parser.add_argument("--model-path", type=Path, help="本机 GGUF 文件；相对当前工作目录")
    parser.add_argument("--slots", type=int, help="服务端槽位上限；V100 先试 2，再试 4")
    parser.add_argument("--print-command", action="store_true", help="仅打印参数，不启动服务")
    args = parser.parse_args()
    settings = load_settings(args.config)
    command = server_command(settings, args.executable, args.model_path, args.slots)
    print(json.dumps(command, ensure_ascii=False))
    if args.print_command:
        return
    if not Path(command[2]).is_file():
        raise FileNotFoundError(f"GGUF 文件不存在：{command[2]}")
    if shutil.which(command[0]) is None:
        raise FileNotFoundError("未找到 llama-server；请配置 llm.llama_cpp.executable 为 CUDA 版可执行文件路径")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(settings.llm.adaptive.gpu_index)
    key = settings.llm.llama_cpp.api_key or environment.get(settings.llm.llama_cpp.api_key_env)
    if key:
        environment["LLAMA_API_KEY"] = key  # Never include credentials in printed commands.
    process = subprocess.Popen(command, env=environment)
    try:
        raise SystemExit(process.wait())
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    main()
