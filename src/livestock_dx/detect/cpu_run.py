"""Run YOLO11x training on a shared laptop without disturbing anything else on it.

  python -m livestock_dx.detect.cpu_run --data data_yolo_weak/dataset.yaml --hours 6
  python -m livestock_dx.detect.cpu_run ... --resume        # continue after a pause / reboot (state is in runs/)

Safeguards (each one is there so the training cannot hurt the machine or the other projects):
  * below-normal CPU priority and a capped thread count: other apps stay responsive
  * memory watchdog: if free RAM drops under --min-free-gb the training process is SUSPENDED (not killed) and
    resumed automatically when RAM recovers; creating the file runs/PAUSE does the same by hand
  * ultralytics is told not to pip-install anything (a tflite export would otherwise downgrade numpy) and to keep its
    settings inside the project instead of %APPDATA%
  * outputs go to runs/ only; datasets are read-only inputs; vetdata/, vetmodels/ and the other apps are never touched
  * every epoch is checkpointed (last.pt), so a kill/reboot loses at most one epoch
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def safe_env(threads: int) -> dict:
    env = dict(os.environ)
    env.update({
        "YOLO_AUTOINSTALL": "False",                          # never pip-install behind our back
        "YOLO_CONFIG_DIR": str(ROOT / ".yolo_config"),        # settings stay in the project, not %APPDATA%
        "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads),
        "PYTHONUNBUFFERED": "1", "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
    })
    return env


def watch(proc, min_free_gb: float, resume_free_gb: float, pause_file: Path, log) -> int:
    import psutil
    p = psutil.Process(proc.pid)
    paused = False
    while proc.poll() is None:
        avail = psutil.virtual_memory().available / 1e9
        want_pause = pause_file.exists() or avail < min_free_gb or (paused and avail < resume_free_gb)
        try:
            kids = p.children(recursive=True)
            if want_pause and not paused:
                for q in [p, *kids]:
                    q.suspend()
                paused = True
                log(f"PAUSED (free RAM {avail:.1f} GB{', PAUSE file present' if pause_file.exists() else ''})")
            elif not want_pause and paused:
                for q in [p, *kids]:
                    q.resume()
                paused = False
                log(f"RESUMED (free RAM {avail:.1f} GB)")
        except psutil.NoSuchProcess:
            break
        time.sleep(5)
    return proc.wait()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--hours", type=float, default=6.0, help="wall-clock budget for the whole schedule")
    ap.add_argument("--threads", type=int, default=max(2, (os.cpu_count() or 8) // 2), help="CPU threads (default: half)")
    ap.add_argument("--min-free-gb", type=float, default=1.0, help="suspend below this much free RAM")
    ap.add_argument("--resume-free-gb", type=float, default=1.8)
    ap.add_argument("--name", default="livestock_cpu")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--task", choices=["detect", "segment"], default="detect")
    ap.add_argument("extra", nargs=argparse.REMAINDER, help="more flags for livestock_dx.detect.train, after --")
    a = ap.parse_args(argv)

    import psutil
    runs = ROOT / "runs"
    runs.mkdir(exist_ok=True)
    logf = open(runs / f"{a.name}.runner.log", "a", encoding="utf-8")
    log = lambda m: (print(m, flush=True), logf.write(f"{time.strftime('%F %T')} {m}\n"), logf.flush())

    cmd = [sys.executable, "-m", "livestock_dx.detect.train", "--data", a.data, "--profile", "cpu", "--device", "cpu",
           "--name", a.name, "--project", str(runs), "--task", a.task] + (["--resume"] if a.resume else
          ["--time", str(a.hours)]) + [x for x in a.extra if x != "--"]
    log(f"start: {' '.join(cmd)}  (threads={a.threads}, budget={a.hours} h, min free RAM {a.min_free_gb} GB)")
    flags = subprocess.BELOW_NORMAL_PRIORITY_CLASS if sys.platform == "win32" else 0
    proc = subprocess.Popen(cmd, cwd=ROOT, env=safe_env(a.threads), creationflags=flags,
                            stdout=open(runs / f"{a.name}.train.log", "a", encoding="utf-8"), stderr=subprocess.STDOUT)
    if sys.platform != "win32":
        psutil.Process(proc.pid).nice(10)
    code = watch(proc, a.min_free_gb, a.resume_free_gb, runs / "PAUSE", log)
    log(f"training process exited with code {code}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
