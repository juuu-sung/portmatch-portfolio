"""Run the frozen comparison from a user terminal; preserve interrupted folds."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import threading

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE))
from portmatch.compound_experiment import check_protocol, read_json, write_json


def complete(path):
    try:
        return read_json(path).get("complete") is True
    except (OSError, ValueError):
        return False


def status(root):
    frozen = check_protocol(root, protection=True)
    return {"teacher_pool_complete": complete(root / "teacher_pool/manifest.json"),
            "folds": {str(f): "complete" if complete(root / f"fold{f}/status.json")
                      else "interrupted" if (root / f"fold{f}").exists() else "not_started"
                      for f in frozen["planned_folds"]},
            "protected_files_verified": len(frozen["protected_files"])}


@contextmanager
def runner_lock(root):
    with (root / "runner.lock").open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0"); stream.flush()
        stream.seek(0)
        if sys.platform == "win32":
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("Another terminal is already running this study") from exc
            try:
                yield
            finally:
                stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def archive_interrupted(root, names):
    """Rename only explicitly checked experiment-local paths; never delete."""
    paths = []
    for name in names:
        path = root / name
        if path.exists():
            resolved = path.resolve()
            if resolved.parent != root or path.is_symlink():
                raise ValueError("Refusing to move a path outside this experiment")
            paths.append(path)
    if not paths:
        return None
    destination = root / ("interrupted_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    if destination.resolve().parent != root:
        raise ValueError("Archive destination escaped experiment")
    destination.mkdir()
    for path in paths:
        path.rename(destination / path.name)
    print(f"Preserved interrupted files: {destination}", flush=True)
    return destination


def run(root, workers):
    active, guard, output_guard, stop = set(), threading.Lock(), threading.Lock(), threading.Event()
    executor = None

    def command(label, args):
        if stop.is_set():
            return 130
        with (root / f"{label}.log").open("w", encoding="utf-8") as log:
            with guard:
                if stop.is_set():
                    return 130
                child = subprocess.Popen([sys.executable, "-u", *args], cwd=WORKSPACE,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                    encoding="utf-8", errors="replace")
                active.add(child)
            try:
                for line in child.stdout:
                    log.write(line); log.flush()
                    with output_guard:
                        print(f"[{label}] {line.rstrip()}", flush=True)
                return child.wait()
            except BaseException:
                if child.poll() is None:
                    child.terminate()
                child.wait()
                raise
            finally:
                child.stdout.close()
                with guard:
                    active.discard(child)

    def cli(*args):
        return ["-m", "portmatch.compound_experiment", *args, "--output", str(root)]

    try:
        state = status(root)
        print(json.dumps(state, indent=2), flush=True)
        if not state["teacher_pool_complete"]:
            archive_interrupted(root, ["teacher_pool", "teacher_pool.log"])
            if command("teacher_pool", cli("prepare-teachers")):
                return 1
        pending = [int(f) for f, s in state["folds"].items() if s != "complete"]
        archive_interrupted(root, [name for f in pending for name in (f"fold{f}", f"fold{f}.log")])
        executor = ThreadPoolExecutor(max_workers=workers)
        futures = {executor.submit(command, f"fold{f}", cli("run", "--fold", str(f))): f for f in pending}
        failed = []
        for future in as_completed(futures):
            fold = futures[future]
            if future.result():
                failed.append(fold)
            write_json(root / "orchestration.json", {"failed_folds": failed,
                "complete_folds": [f for f in range(5) if complete(root / f"fold{f}/status.json")]})
        if failed:
            print(f"Incomplete folds: {failed}. Inspect fold logs; no partial results promoted.")
            return 1
        if command("comparison", cli("report")):
            return 1
        if command("analysis", [str(WORKSPACE / "scripts/analyze_compound.py"), "--root", str(root)]):
            return 1
        print(f"Completed comparison and audit: {root / 'metrics.csv'}", flush=True)
        return 0
    except KeyboardInterrupt:
        stop.set()
        with guard:
            for child in active:
                if child.poll() is None:
                    child.terminate()
        print("Stopped. Completed stages are reusable; interrupted fold files remain preserved.", flush=True)
        return 130
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=3, choices=range(1, 4))
    parser.add_argument("--status", action="store_true", help="Read status only; do not train or move files")
    args = parser.parse_args()
    root = args.output.resolve()
    if args.status:
        print(json.dumps(status(root), indent=2))
        return 0
    with runner_lock(root):
        return run(root, args.workers)


if __name__ == "__main__":
    sys.exit(main())
