"""Trainer callback for multi-provider fleet sync."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from transformers import TrainerCallback

from src.fleet.env import FleetEnv
from src.fleet.store import LocalFleetStore


class FleetSyncCallback(TrainerCallback):
    """Record heartbeats and optionally reload canonical weights after a fleet merge."""

    def __init__(self, fleet: FleetEnv, output_dir: Path, store: Optional[LocalFleetStore] = None):
        self.fleet = fleet
        self.output_dir = Path(output_dir)
        self.store = store
        self.fleet_dir = self.output_dir / "fleet"
        self.fleet_dir.mkdir(parents=True, exist_ok=True)
        self._pending_reload = False

    def _write_worker_heartbeat(self, step: int, checkpoint: Optional[Path]) -> None:
        payload = {
            "worker_id": self.fleet.worker_id,
            "step": step,
            "checkpoint": str(checkpoint) if checkpoint else None,
            "ts": time.time(),
        }
        (self.fleet_dir / "heartbeat.json").write_text(json.dumps(payload, indent=2))
        if self.store is not None:
            self.store.write_heartbeat(
                self.fleet.worker_id,
                step=step,
                checkpoint=str(checkpoint) if checkpoint else None,
            )
            if checkpoint and checkpoint.is_dir():
                self.store.publish_worker_checkpoint(self.fleet.worker_id, checkpoint, step)

    def _canonical_path(self) -> Optional[Path]:
        if self.store is not None:
            state = self.store.load_state()
            if not self.store.reload_ready(state.sync_generation):
                return None
            return self.store.canonical_dir()
        reload_path = self.fleet_dir / "reload.json"
        if reload_path.is_file():
            raw = json.loads(reload_path.read_text()).get("canonical_path")
            return Path(raw) if raw else None
        return None

    def _reload_canonical_weights(self, model) -> None:
        canonical = self._canonical_path()
        if canonical is None or not canonical.is_dir():
            return
        print(f"[fleet] Reloading canonical weights from {canonical}")
        from transformers import AutoModelForCausalLM

        reloaded = AutoModelForCausalLM.from_pretrained(str(canonical))
        model.load_state_dict(reloaded.state_dict(), strict=False)
        del reloaded
        self._pending_reload = False

    def on_save(self, args, state, control, **kwargs):
        ckpt = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        self._write_worker_heartbeat(state.global_step, ckpt if ckpt.is_dir() else None)

        if (
            self.fleet.sync_mode == "continuous"
            and self.fleet.sync_every_steps > 0
            and state.global_step > 0
            and state.global_step % self.fleet.sync_every_steps == 0
        ):
            self._pending_reload = True

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if self._pending_reload and model is not None:
            self._reload_canonical_weights(model)


def build_fleet_callback(cfg: Dict[str, Any], output_dir: Path) -> Optional[FleetSyncCallback]:
    fleet = FleetEnv.from_os()
    if not fleet.enabled:
        return None

    store = None
    if fleet.store_root:
        store = LocalFleetStore(fleet.run_name, base_dir=Path(fleet.store_root))
    elif fleet.sync_mode == "continuous":
        store = LocalFleetStore(fleet.run_name)

    print(
        f"[fleet] worker={fleet.worker_id} rank={fleet.rank}/{fleet.world_size} "
        f"sync_every={fleet.sync_every_steps} mode={fleet.sync_mode}"
    )
    return FleetSyncCallback(fleet=fleet, output_dir=output_dir, store=store)
