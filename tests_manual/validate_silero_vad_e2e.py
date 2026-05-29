"""Silero-VAD Phase-3-bundle end-to-end validation (CPU).

Validates the Track 19 WORKER_ENV migration + T24 meta cleanup live, mirroring the
Voxtral-HF Phase 3 validation pattern (project-local runtime, PluginManager + JobQueue).
Silero-VAD is CPU-only with bundled wheel weights — no GPU monitor, no model download,
no heartbeat — so this asserts the migrated manifest shape and runs a real VAD analysis
through a spawned worker (proving the WORKER_ENV overlay is composed + injected at Popen).

Run from the silero-vad repo root after:

  1. `cjm-ctl --cjm-config cjm.yaml setup-runtime`
  2. `cjm-ctl --cjm-config cjm.yaml install-all --plugins plugins_test.yaml`
  3. A short audio clip at test_files/short_test_audio.mp3

Then:

  conda run -n cjm-media-plugin-silero-vad --no-capture-output \\
    python tests_manual/validate_silero_vad_e2e.py

This script:
  - Verifies the silero-vad v2.0 manifest carries (a) a non-empty `description`
    (substrate validator requirement / V1 gate), (b) Phase-5a binary-only resources
    (no quantitative min_*_mb fields), and (c) the Track 19 `worker_env` with a STATIC
    OMP_NUM_THREADS default + an empty install.env_vars (the var moved onto the class).
  - Runs a real VAD analysis via JobQueue.submit -> wait_for_job through a spawned worker
    and asserts speech ranges were detected.
  - Logs any empirical_resources.db rows (CPU plugin: memory recorded, gpu peak 0).
"""
import json
import logging
import sqlite3
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s",
)
log = logging.getLogger("silero-vad-e2e")

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_AUDIO = REPO_ROOT / "test_files" / "short_test_audio.mp3"
MANIFESTS_DIR = REPO_ROOT / ".cjm" / "manifests"
EMPIRICAL_DB = REPO_ROOT / ".cjm" / "empirical_resources.db"

PLUGIN_NAME = "cjm-media-plugin-silero-vad"


def check_prereqs() -> None:
    assert TEST_AUDIO.exists(), f"Missing test audio: {TEST_AUDIO}"
    assert MANIFESTS_DIR.exists(), (
        f"Missing manifests dir: {MANIFESTS_DIR} — run cjm-ctl setup-runtime + install-all first"
    )
    assert (MANIFESTS_DIR / f"{PLUGIN_NAME}.json").exists(), f"Missing manifest: {PLUGIN_NAME}.json"
    log.info("Prereqs OK: test audio + silero-vad manifest present")


def assert_manifest_shape() -> None:
    """v2.0 manifest must carry the T24 description, Phase-5a binary-only resources,
    and the Track 19 worker_env (OMP_NUM_THREADS static default) with empty
    install.env_vars."""
    manifest = json.loads((MANIFESTS_DIR / f"{PLUGIN_NAME}.json").read_text())
    assert manifest["format_version"] == "2.0", manifest["format_version"]
    code = manifest["code"]

    # T24: non-empty description required by the substrate validator (SG-6 / V1 gate).
    desc = code.get("description") or manifest.get("description") or ""
    assert desc.strip(), "manifest description is empty (T24 regression)"
    log.info(f"Manifest T24 description: {desc!r}")

    # CR-1 taxonomy + Phase 5a resources.
    tax = code["taxonomy"]
    assert tax["domain"] == "media" and tax["role"] == "MediaAnalysisPlugin", tax
    assert code["resources"]["requires_gpu"] is False, code["resources"]
    # CR-7 reframe / V12 gate: no quantitative resource fields should remain.
    for stale in ("min_gpu_vram_mb", "recommended_gpu_vram_mb", "min_system_ram_mb"):
        assert stale not in code["resources"], f"stale resource field present: {stale}"
    log.info(f"Manifest CR-1/Phase-5a: taxonomy={tax}, resources={code['resources']}")

    # Track 19: WORKER_ENV migrated from install.env_vars. OMP_NUM_THREADS is a
    # STATIC default (no ${...} template).
    worker_env = code.get("worker_env", [])
    by_name = {e["name"]: e for e in worker_env}
    assert "OMP_NUM_THREADS" in by_name, (
        f"Track 19 WORKER_ENV missing OMP_NUM_THREADS: {sorted(by_name)}"
    )
    omp_default = by_name["OMP_NUM_THREADS"].get("default", "")
    assert omp_default == "4", f"OMP_NUM_THREADS default unexpected: {omp_default!r}"
    # install.env_vars must be empty post-migration (the var moved onto the class).
    install_env = manifest.get("install", {}).get("env_vars", {})
    assert not install_env, f"install.env_vars should be empty post-migration: {install_env}"
    log.info(
        f"Manifest Track 19 worker_env: {sorted(by_name)} | "
        f"OMP_NUM_THREADS default={omp_default!r}; install.env_vars empty"
    )


def run_e2e() -> None:
    """Run a real VAD analysis through a spawned worker via the JobQueue."""
    import asyncio

    from cjm_plugin_system.core.manager import PluginManager
    from cjm_plugin_system.core.config import get_config
    from cjm_plugin_system.core.queue import JobQueue, JobStatus

    cfg = get_config()
    log.info(f"data_dir={cfg.data_dir}, manifests_dir={cfg.manifests_dir}")

    pm = PluginManager(search_paths=[MANIFESTS_DIR])
    pm.discover_manifests()
    log.info(f"Discovered: {[m.name for m in pm.discovered]}")

    silero_meta = next(m for m in pm.discovered if m.name == PLUGIN_NAME)
    ok = pm.load_plugin(silero_meta, config={})
    assert ok, f"Failed to load {PLUGIN_NAME}"
    silero_id = silero_meta.name
    log.info(f"Loaded {PLUGIN_NAME} as instance_id={silero_id}")

    async def run_job():
        queue = JobQueue(deps=pm)
        await queue.start()
        try:
            job_id = await queue.submit(silero_id, str(TEST_AUDIO))
            log.info(f"Submitted VAD job {job_id} for {TEST_AUDIO.name}")
            job = await queue.wait_for_job(job_id, timeout=600)
            if job.status != JobStatus.completed:
                raise RuntimeError(f"job {job_id} status={job.status} error={job.error}")
            return job.result
        finally:
            await queue.stop()

    log.info("Submitting VAD job via JobQueue...")
    t0 = time.time()
    result = asyncio.run(run_job())

    # Result crosses the worker IPC boundary; accept dict-or-object.
    ranges = result.get("ranges") if isinstance(result, dict) else getattr(result, "ranges", None)
    meta = result.get("metadata") if isinstance(result, dict) else getattr(result, "metadata", {})
    log.info(f"VAD completed in {time.time() - t0:.1f}s")
    assert ranges, f"No speech ranges detected; raw result={result!r}"
    log.info(f"Detected {len(ranges)} speech segments; metadata={meta}")
    for i, r in enumerate(ranges[:3]):
        rd = r if isinstance(r, dict) else {"start": getattr(r, "start", None), "end": getattr(r, "end", None)}
        log.info(f"  [{i}] {rd.get('start')}s - {rd.get('end')}s")

    # Empirical store: CPU plugin -> expect a row with memory recorded (gpu peak 0).
    # Informational only (not a hard gate for a CPU plugin).
    if EMPIRICAL_DB.exists():
        try:
            con = sqlite3.connect(EMPIRICAL_DB)
            try:
                tables = [r[0] for r in con.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
                for t in tables:
                    cols = [r[1] for r in con.execute(f"PRAGMA table_info({t})").fetchall()]
                    rows = con.execute(
                        f"SELECT * FROM {t} WHERE plugin_name=? OR instance_id=? OR instance_id LIKE ?",
                        (PLUGIN_NAME, silero_id, f"{PLUGIN_NAME}%"),
                    ).fetchall()
                    for r in rows:
                        log.info(f"  empirical {t}: {dict(zip(cols, r))}")
            finally:
                con.close()
        except Exception as e:
            log.warning(f"empirical store inspection skipped: {e}")
    else:
        log.info("(no empirical store yet — fine for a single CPU run)")

    pm.unload_plugin(silero_id)
    log.info("Unloaded plugin; validation done.")


def main() -> int:
    check_prereqs()
    assert_manifest_shape()
    run_e2e()
    log.info("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
