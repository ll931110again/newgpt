from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

from src.data.packed_dataset import Manifest
from src.fleet.config import load_fleet_config
from src.fleet.merge import fedavg_checkpoints
from src.fleet.providers import build_provider
from src.fleet.shards import assign_shards, reassign_shards_after_failure
from src.fleet.state import FleetState, WorkerRecord
from src.fleet.store import LocalFleetStore, fleet_store_from_config
from src.utils.config import load_yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _count_shards(train_config: str, repo_root: Path) -> int:
    cfg = load_yaml(str(repo_root / train_config))
    manifest_path = cfg.get("data", {}).get("dataset_manifest")
    if not manifest_path:
        return 1
    path = repo_root / manifest_path
    if not path.is_file():
        return 1
    return len(Manifest.load(str(path)).shards)


def _worker_env(state: FleetState, worker: WorkerRecord, train_config: str, repo_root: Path) -> Dict[str, str]:
    bundled = repo_root / "infra/kaggle/dataset/checkpoints/canonical"
    if (bundled / "model.safetensors").is_file() or (bundled / "pytorch_model.bin").is_file():
        canonical = "checkpoints/canonical"
    else:
        canonical = state.canonical_path or ""
    sync_mode = "session" if worker.provider_type == "kaggle" else "continuous"
    return {
        "FLEET_ENABLED": "1",
        "FLEET_WORKER_ID": worker.worker_id,
        "FLEET_RUN_NAME": state.run_name,
        "FLEET_RANK": str(worker.rank),
        "FLEET_WORLD_SIZE": str(state.world_size),
        "FLEET_SHARD_IDS": ",".join(str(i) for i in worker.shard_ids),
        "FLEET_SYNC_EVERY_STEPS": str(state.sync_every_steps),
        "FLEET_SYNC_MODE": sync_mode,
        "FLEET_TRAIN_CONFIG": train_config,
        "FLEET_CANONICAL_INIT": canonical,
        "FLEET_STORE_ROOT": str(Path("runs/fleet")),
    }


class FleetOrchestrator:
    def __init__(self, fleet_cfg_path: str):
        self.repo_root = _repo_root()
        self.fleet_cfg = load_fleet_config(fleet_cfg_path)
        self.store: LocalFleetStore = fleet_store_from_config(self.fleet_cfg)  # type: ignore[assignment]

    def _enabled_providers(self) -> List[Dict]:
        return [p for p in self.fleet_cfg["providers"] if p.get("enabled", True)]

    def init_state(self) -> FleetState:
        providers = self._enabled_providers()
        if not providers:
            raise SystemExit("No enabled providers in fleet config")

        train_config = str(self.fleet_cfg["train_config"])
        total_shards = _count_shards(train_config, self.repo_root)
        workers: List[WorkerRecord] = []
        for rank, spec in enumerate(providers):
            wid = str(spec["id"])
            shard_ids = assign_shards(total_shards, rank, len(providers))
            workers.append(
                WorkerRecord(
                    worker_id=wid,
                    provider_type=str(spec["type"]),
                    rank=rank,
                    shard_ids=shard_ids,
                    status="pending",
                )
            )
        return self.store.init_run(
            train_config=train_config,
            sync_every_steps=int(self.fleet_cfg["sync_every_steps"]),
            sync_timeout_minutes=int(self.fleet_cfg.get("sync_timeout_minutes", 30)),
            workers=workers,
            total_shards=total_shards,
        )

    def start(self) -> None:
        state_path = self.store.state_path()
        if state_path.is_file():
            state = self.store.load_state()
        else:
            state = self.init_state()

        train_config = state.train_config
        for worker in state.workers:
            if worker.status == "dead":
                continue
            provider = build_provider(
                worker.worker_id,
                worker.provider_type,
                self._provider_cfg(worker.worker_id),
                self.repo_root,
            )
            env = _worker_env(state, worker, train_config, self.repo_root)
            provider.launch(env)
            worker.status = "running"
            worker.last_heartbeat_ts = time.time()
        self.store.save_state(state)
        print(f"[fleet] Launched {len(state.workers)} workers for run {state.run_name}")

    def status(self) -> None:
        state = self.store.load_state()
        print(json.dumps(state.to_dict(), indent=2))
        for worker in state.workers:
            provider = build_provider(
                worker.worker_id,
                worker.provider_type,
                self._provider_cfg(worker.worker_id),
                self.repo_root,
            )
            st = provider.status()
            print(f"[fleet] {worker.worker_id}: {st.phase} {st.detail[:120]}")

    def _provider_cfg(self, worker_id: str) -> Dict:
        for spec in self.fleet_cfg["providers"]:
            if spec["id"] == worker_id:
                return dict(spec)
        return {}

    def sync(self) -> None:
        state = self.store.load_state()
        for worker in state.workers:
            if worker.status == "dead":
                continue
            provider = build_provider(
                worker.worker_id,
                worker.provider_type,
                self._provider_cfg(worker.worker_id),
                self.repo_root,
            )
            dest = self.store.worker_checkpoint_dir(worker.worker_id)
            ckpt = provider.fetch_latest_checkpoint(dest)
            if ckpt is not None:
                worker.last_checkpoint = str(ckpt)
                hb = self.store.read_heartbeat(worker.worker_id)
                if hb:
                    worker.last_step = int(hb.get("step", worker.last_step))
                    worker.last_heartbeat_ts = float(hb.get("ts", time.time()))
        self.store.save_state(state)
        print("[fleet] Sync complete")

    def merge(self, min_workers: int = 1) -> Optional[Path]:
        state = self.store.load_state()
        ckpt_dirs: List[Path] = []
        for worker in state.workers:
            if worker.status == "dead":
                continue
            if worker.last_checkpoint:
                p = Path(worker.last_checkpoint)
                if p.is_dir():
                    ckpt_dirs.append(p)
            else:
                latest = self.store.list_worker_checkpoints(worker.worker_id)
                if latest:
                    ckpt_dirs.append(latest[-1])

        if len(ckpt_dirs) < min_workers:
            print(f"[fleet] Need >={min_workers} checkpoints, got {len(ckpt_dirs)}")
            return None

        state.sync_generation += 1
        out = self.store.canonical_dir() / f"checkpoint-sync-{state.sync_generation}"
        fedavg_checkpoints(ckpt_dirs, out)
        state.canonical_path = str(out)
        state.canonical_step = max(w.last_step for w in state.workers)
        self.store.write_reload_signal(state.sync_generation)
        self.store.save_state(state)
        print(f"[fleet] Merged {len(ckpt_dirs)} checkpoints -> {out}")
        return out

    def detect_stale(self, stale_seconds: float = 7200.0) -> None:
        state = self.store.load_state()
        now = time.time()
        dead_ids = []
        for worker in state.workers:
            if worker.status not in {"running", "pending"}:
                continue
            if worker.last_heartbeat_ts and now - worker.last_heartbeat_ts > stale_seconds:
                worker.status = "dead"
                dead_ids.append(worker.worker_id)
        if dead_ids:
            assignments = reassign_shards_after_failure(
                state.total_shards,
                [asdict(w) for w in state.workers],
            )
            for worker in state.workers:
                worker.shard_ids = assignments.get(worker.worker_id, [])
            print(f"[fleet] Marked dead: {dead_ids}; reassigned shards")
        self.store.save_state(state)

    def resume(self) -> None:
        state = self.store.load_state()
        train_config = state.train_config
        relaunched = 0
        for worker in state.workers:
            if worker.status != "dead":
                continue
            provider = build_provider(
                worker.worker_id,
                worker.provider_type,
                self._provider_cfg(worker.worker_id),
                self.repo_root,
            )
            env = _worker_env(state, worker, train_config, self.repo_root)
            provider.launch(env)
            worker.status = "running"
            worker.last_heartbeat_ts = time.time()
            relaunched += 1
        self.store.save_state(state)
        print(f"[fleet] Relaunched {relaunched} dead workers")

    def restart(self) -> None:
        """Stage latest checkpoint, reset fleet state, and launch all workers."""
        import subprocess

        stage = self.repo_root / "scripts" / "stage_fleet_checkpoint.sh"
        subprocess.check_call([str(stage)], cwd=str(self.repo_root))
        canonical_rel = "checkpoints/canonical"
        fleet_canon = self.store.canonical_dir() / "checkpoint-latest"
        state = self.init_state()
        state.sync_generation = 0
        state.canonical_step = 0
        state.canonical_path = str(fleet_canon) if fleet_canon.is_dir() else canonical_rel
        self.store.save_state(state)
        print(f"[fleet] Restarting from canonical: {state.canonical_path}")
        self.start()


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-provider fleet orchestrator")
    ap.add_argument("--spec", required=True, help="Path to infra/fleet/fleet.yaml")
    ap.add_argument(
        "command",
        choices=["start", "status", "sync", "merge", "resume", "restart", "watch"],
    )
    args = ap.parse_args()
    orch = FleetOrchestrator(args.spec)

    if args.command == "start":
        orch.start()
    elif args.command == "status":
        orch.status()
    elif args.command == "sync":
        orch.sync()
    elif args.command == "merge":
        orch.merge()
    elif args.command == "resume":
        orch.resume()
    elif args.command == "restart":
        orch.restart()
    elif args.command == "watch":
        interval = int(orch.fleet_cfg.get("watch_interval_seconds", 120))
        while True:
            orch.sync()
            orch.detect_stale()
            orch.merge(min_workers=1)
            time.sleep(interval)


if __name__ == "__main__":
    main()
