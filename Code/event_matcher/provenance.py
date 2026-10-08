"""Tie every real run to the committed files actually used for the matcher."""
import hashlib
import json
import subprocess
from pathlib import Path


def code_provenance():
    code=Path(__file__).resolve().parents[1]
    root=code.parent
    paths=sorted([*code.glob("*.py"),*code.glob("*.ps1"),*code.joinpath("event_matcher").glob("*.py"),code/"requirements.txt"])
    files={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.exists()}
    digest=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    try:
        commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True,stderr=subprocess.DEVNULL,timeout=5).strip()
        dirty=subprocess.check_output(["git","diff","--name-only","HEAD","--",*files],cwd=root,text=True,stderr=subprocess.DEVNULL,timeout=5).splitlines()
        untracked=subprocess.check_output(["git","ls-files","--others","--exclude-standard","--",*files],cwd=root,text=True,stderr=subprocess.DEVNULL,timeout=5).splitlines()
    except (OSError,subprocess.SubprocessError):
        commit=None;dirty=[];untracked=[]
    return {"git_commit":commit,"code_sha256":digest,"files":files,"current_code_dirty":bool(dirty or untracked),
            "note":"仅检查当前实现文件；不将工作区已有的其他文件删除纳入本轮提交"}
