"""Verify the actual ZIP in isolated processes on public held-out inputs.

No training, cache loading or hidden evaluation inputs are used. Each case gets
its own extracted package and official data/output layout. This is a local OS
measurement, not a guarantee about the unpublished server test size.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import subprocess
import sys
import time
import zipfile

import pandas as pd


QUERY_COLUMNS = ["sample_id", "vessel_name", "call_sign", "gross_tonnage",
                 "international_gross_tonnage", "flag_name", "vessel_type", "arrival_year"]


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def extract_archive(archive, destination):
    with zipfile.ZipFile(archive) as zipped:
        entries = zipped.infolist()
        if len({entry.filename for entry in entries}) != len(entries):
            raise ValueError("ZIP contains duplicate entries")
        if sum(entry.file_size for entry in entries) > 32_000_000_000:
            raise ValueError("ZIP exceeds the expanded-size budget")
        for entry in entries:
            path = PurePosixPath(entry.filename)
            if (path.is_absolute() or ".." in path.parts or "\\" in entry.filename
                    or not path.parts or ":" in entry.filename):
                raise ValueError("ZIP contains an unsafe path")
        if {PurePosixPath(e.filename).parts[0] for e in entries} != {"script.py", "requirements.txt", "model"}:
            raise ValueError("ZIP top-level structure differs from the competition interface")
        zipped.extractall(destination)
    return sum(e.file_size for e in entries)


class ProcessMemoryMonitor:
    """Track the interpreter behind a Windows venv launcher as well as its parent.

    Windows' venv python.exe launches another python.exe. Measuring only Popen's
    handle reports the tiny launcher, not inference. Handles are retained until
    close so a discovered child's native high-water counter remains queryable
    even if it exits between samples. The maximum individual native peak and the
    sampled simultaneous tree RSS are separate metrics, not a sum of peaks.
    """

    def __init__(self, pid):
        self.pid = int(pid)
        self.peak_rss_bytes = 0
        self.sampled_tree_peak_rss_bytes = 0
        self.handles = {}
        self.processes = {}
        if os.name != "nt":
            return
        from ctypes import wintypes as w

        class Entry(ctypes.Structure):
            _fields_ = [("dwSize", w.DWORD), ("cntUsage", w.DWORD),
                        ("th32ProcessID", w.DWORD), ("th32DefaultHeapID", ctypes.c_size_t),
                        ("th32ModuleID", w.DWORD), ("cntThreads", w.DWORD),
                        ("th32ParentProcessID", w.DWORD), ("pcPriClassBase", w.LONG),
                        ("dwFlags", w.DWORD), ("szExeFile", w.WCHAR * 260)]

        class Memory(ctypes.Structure):
            _fields_ = [("cb", w.DWORD), ("PageFaultCount", w.DWORD),
                        *[(name, ctypes.c_size_t) for name in (
                            "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                            "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                            "PagefileUsage", "PeakPagefileUsage", "PrivateUsage")]]

        self.Entry, self.Memory = Entry, Memory
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.psapi = ctypes.WinDLL("psapi", use_last_error=True)
        self.kernel.CreateToolhelp32Snapshot.argtypes = [w.DWORD, w.DWORD]
        self.kernel.CreateToolhelp32Snapshot.restype = w.HANDLE
        for name in ("Process32FirstW", "Process32NextW"):
            function = getattr(self.kernel, name)
            function.argtypes, function.restype = [w.HANDLE, ctypes.POINTER(Entry)], w.BOOL
        self.kernel.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
        self.kernel.OpenProcess.restype = w.HANDLE
        self.kernel.CloseHandle.argtypes, self.kernel.CloseHandle.restype = [w.HANDLE], w.BOOL
        self.psapi.GetProcessMemoryInfo.argtypes = [w.HANDLE, ctypes.POINTER(Memory), w.DWORD]
        self.psapi.GetProcessMemoryInfo.restype = w.BOOL

    def _discover(self):
        snapshot = self.kernel.CreateToolhelp32Snapshot(2, 0)  # TH32CS_SNAPPROCESS
        if snapshot == ctypes.c_void_p(-1).value:
            return
        entries = {}
        try:
            entry = self.Entry()
            entry.dwSize = ctypes.sizeof(entry)
            present = self.kernel.Process32FirstW(snapshot, ctypes.byref(entry))
            while present:
                entries[int(entry.th32ProcessID)] = (int(entry.th32ParentProcessID), entry.szExeFile)
                present = self.kernel.Process32NextW(snapshot, ctypes.byref(entry))
        finally:
            self.kernel.CloseHandle(snapshot)
        descendants = {self.pid, *self.handles}
        while True:
            expanded = descendants | {pid for pid, (parent, _) in entries.items() if parent in descendants}
            if expanded == descendants:
                break
            descendants = expanded
        for pid in descendants - self.handles.keys():
            handle = self.kernel.OpenProcess(0x0400 | 0x0010, False, pid)  # query + VM read only
            if handle:
                self.handles[pid] = handle
                self.processes[pid] = {"pid": pid, "parent_pid": entries.get(pid, (None, None))[0],
                                       "executable": entries.get(pid, (None, None))[1],
                                       "native_peak_rss_bytes": 0}

    def sample(self):
        if os.name == "nt":
            self._discover()
            current_tree = 0
            for pid, handle in self.handles.items():
                memory = self.Memory()
                memory.cb = ctypes.sizeof(memory)
                if self.psapi.GetProcessMemoryInfo(handle, ctypes.byref(memory), memory.cb):
                    peak = int(memory.PeakWorkingSetSize)
                    self.processes[pid]["native_peak_rss_bytes"] = max(
                        self.processes[pid]["native_peak_rss_bytes"], peak)
                    self.peak_rss_bytes = max(self.peak_rss_bytes, peak)
                    current_tree += int(memory.WorkingSetSize)
            self.sampled_tree_peak_rss_bytes = max(self.sampled_tree_peak_rss_bytes, current_tree)
            return self.peak_rss_bytes
        try:
            for line in Path(f"/proc/{self.pid}/status").read_text().splitlines():
                if line.startswith("VmHWM:"):
                    self.peak_rss_bytes = max(self.peak_rss_bytes, int(line.split()[1]) * 1024)
        except (FileNotFoundError, PermissionError):
            pass
        return self.peak_rss_bytes

    def report(self):
        return {"peak_rss_bytes": self.peak_rss_bytes or None,
                "peak_rss_measurement": ("Largest native PeakWorkingSetSize among launcher and descendants."
                                         if os.name == "nt" else "Linux parent process VmHWM."),
                "sampled_process_tree_peak_rss_bytes": self.sampled_tree_peak_rss_bytes or None,
                "memory_processes": list(self.processes.values())}

    def close(self):
        for handle in self.handles.values():
            self.kernel.CloseHandle(handle)
        self.handles.clear()


def terminate_owned_process_tree(process, memory_monitor):
    """Stop only this run's launcher and discovered descendants on an error.

    Retained query handles pin process identities while termination handles are
    opened, so an unrelated process cannot be selected through PID reuse.
    """
    memory_monitor.sample()
    if process.poll() is None:
        process.kill()
    if os.name == "nt":
        from ctypes import wintypes as w
        kernel = memory_monitor.kernel
        kernel.TerminateProcess.argtypes, kernel.TerminateProcess.restype = [w.HANDLE, w.UINT], w.BOOL
        kernel.WaitForSingleObject.argtypes, kernel.WaitForSingleObject.restype = [w.HANDLE, w.DWORD], w.DWORD
        for pid in memory_monitor.handles:
            if pid == process.pid:
                continue
            handle = kernel.OpenProcess(0x0001 | 0x00100000, False, pid)  # terminate + synchronize
            if handle:
                try:
                    kernel.TerminateProcess(handle, 1)
                    kernel.WaitForSingleObject(handle, 5000)
                finally:
                    kernel.CloseHandle(handle)
    process.wait()


def run_case(archive, root, label, rows, catalog, python, timeout):
    case = root / label
    case.mkdir()
    expanded_bytes = extract_archive(archive, case)
    (case / "data").mkdir()
    rows[QUERY_COLUMNS].to_csv(case / "data/test.csv", index=False, encoding="utf-8-sig")
    shutil.copyfile(catalog, case / "data/vessel_catalog.csv")
    before = {name: file_hash(case / "data" / name) for name in ("test.csv", "vessel_catalog.csv")}
    log_path = case / "execution.log"
    start = time.perf_counter()
    peak = 0
    print(f"Running {label}: {len(rows):,} public input rows", flush=True)
    with log_path.open("wb") as log:
        process = subprocess.Popen([str(python), "-I", str(case / "script.py")],
                                   cwd=case, stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        memory_monitor = ProcessMemoryMonitor(process.pid)
        try:
            while process.poll() is None:
                peak = memory_monitor.sample()
                if time.perf_counter() - start > timeout:
                    terminate_owned_process_tree(process, memory_monitor)
                    raise TimeoutError(f"{label} exceeded {timeout} seconds; see its execution log")
                time.sleep(.1)
            peak = memory_monitor.sample()
        finally:
            if process.poll() is None:
                terminate_owned_process_tree(process, memory_monitor)
            memory_report = memory_monitor.report()
            memory_monitor.close()
    seconds = time.perf_counter() - start
    if process.returncode:
        raise RuntimeError(f"{label} failed with exit {process.returncode}; see {log_path}")
    after = {name: file_hash(case / "data" / name) for name in before}
    if before != after:
        raise AssertionError("Submission changed official input files")
    output = pd.read_csv(case / "output/submission.csv", dtype=str, keep_default_na=False)
    if (output.columns.tolist() != ["sample_id", "target_vessel_id"]
            or output.sample_id.tolist() != rows.sample_id.tolist()
            or not output.sample_id.is_unique
            or output.target_vessel_id.eq("").any()):
        raise AssertionError("Submission output schema, order or identities disagree with input")
    result = {"case": label, "rows": len(rows), "wall_seconds": seconds,
              "wall_rows_per_second": len(rows) / seconds, **memory_report,
              "expanded_package_bytes": expanded_bytes, "input_files_unchanged": True,
              "output_sha256": file_hash(case / "output/submission.csv"), "exit_code": process.returncode}
    # The script emits aggregate timing only, never private input or prediction rows.
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            result.setdefault("script_events", []).append(value)
    print(f"Completed {label}: {seconds:.2f}s, peak RSS {peak / 1024**2:.1f} MiB", flush=True)
    return result, output.set_index("sample_id").target_vessel_id.to_dict()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", required=True, dest="archive", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train", default="work/data/train.csv", type=Path)
    parser.add_argument("--catalog", default="work/data/vessel_catalog.csv", type=Path)
    parser.add_argument("--reference-model", default="artifacts/ranker_token_phonetic_fold0", type=Path)
    parser.add_argument("--python", default=sys.executable, type=Path)
    parser.add_argument("--timeout", default=1800, type=int)
    args = parser.parse_args()
    root, archive = args.output.resolve(), args.archive.resolve()
    if root.exists():
        raise FileExistsError("Use a new verification output directory")
    if args.timeout < 1:
        raise ValueError("Timeout must be positive")
    queries = pd.read_csv(args.train, encoding="utf-8-sig", dtype={"sample_id": str,
                           "call_sign": str, "vessel_name": str, "target_vessel_id": str})
    reference = pd.read_csv(args.reference_model / "validation_predictions.csv", dtype={
                            "sample_id": str, "predicted_vessel_id": str, "target_vessel_id": str})
    if not reference.sample_id.is_unique or not queries.sample_id.is_unique:
        raise ValueError("Verification input or reference IDs are duplicated")
    heldout = queries.set_index("sample_id").loc[reference.sample_id].reset_index()
    if heldout.target_vessel_id.tolist() != reference.target_vessel_id.tolist():
        raise ValueError("Public query labels disagree with reference provenance")
    languages = reference.has_hangul.astype(str).str.lower().isin(["true", "1"])
    positions = [*reference.index[languages][:32], *reference.index[~languages][:32]]
    subset = heldout.iloc[positions]
    catalog_ids = set(pd.read_csv(args.catalog, usecols=["vessel_id"], dtype=str).vessel_id)
    expected = reference.set_index("sample_id").predicted_vessel_id.to_dict()
    root.mkdir(parents=True)
    result = {"complete": False, "archive": str(archive), "archive_sha256": file_hash(archive),
              "archive_bytes": archive.stat().st_size, "reference_model": str(args.reference_model.resolve()),
              "platform": platform.platform(), "python": platform.python_version(),
              "processor": os.environ.get("PROCESSOR_IDENTIFIER", platform.processor()),
              "logical_cpus_visible": os.cpu_count(), "lightgbm_threads": 6,
              "source": "Public train.csv rows held out by the fixed fold0 model; no feature cache used.",
              "rows": len(heldout), "hangul_rows": int(languages.sum()),
              "cases": [], "limits": ["Windows/local OS measurement, not DACON Linux validation.",
              "Hidden test size is unpublished; local timing does not guarantee the 30-minute limit.",
              "Each case is a fresh process; OS filesystem caches were not cleared."]}

    def save():
        (root / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    save()
    full = None
    for label, rows in (("full", heldout), ("reversed", heldout.iloc[::-1]),
                        ("subset", subset), ("single", subset.iloc[:1])):
        measured, predictions = run_case(archive, root, label, rows, args.catalog.resolve(),
                                         args.python.resolve(), args.timeout)
        if not set(predictions.values()).issubset(catalog_ids):
            raise AssertionError("Predicted vessel is outside the official catalog")
        if predictions != {sid: expected[sid] for sid in rows.sample_id}:
            raise AssertionError("ZIP predictions differ from the saved 86-feature model")
        if full is None:
            full = predictions
        if predictions != {sid: full[sid] for sid in rows.sample_id}:
            raise AssertionError("Predictions depend on order, subset or batch size")
        measured.update(matches_reference=True, matches_full_by_id=True, catalog_ids_valid=True)
        result["cases"].append(measured)
        save()
    if file_hash(archive) != result["archive_sha256"]:
        raise AssertionError("Submission ZIP changed during verification")
    result.update(complete=True, row_independence_verified=True,
                  all_predictions_match_saved_model=True, upload_performed=False)
    save()
    print(f"Verified ZIP and all four input modes. Report: {root / 'verification.json'}", flush=True)


if __name__ == "__main__":
    main()
