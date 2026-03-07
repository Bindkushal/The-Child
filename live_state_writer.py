"""
live_state_writer.py
====================
Writes real-time training state to live_state.json.
brain.html polls /api/state every 500ms — this is the data source.

SCHEMA expected by brain.html:
  current  : step, task_id, n_tasks, loss, ewc_loss, acc, running
  network  : input_size, output_size, total_params, layers[{id,type,size,nes}]
  history  : [{step, loss, ewc, acc, params}]   (rolling 200-point window)
  recent_events : [{type, layer, count, step, trigger}]

Usage in train.py:
  writer = LiveStateWriter("live_state.json", write_every=5)

  # inside training loop:
  writer.update(
      step=global_step, task_id=task_id, n_tasks=CONFIG["n_tasks"],
      loss=loss_val, ewc_loss=ewc_penalty.item(), acc=acc,
      net=net, controller=controller,
      new_events=new_events   # events from THIS step only
  )

  # at the very end:
  writer.finish(...)
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path


class LiveStateWriter:
    def __init__(self, path: str = "live_state.json",
                 write_every: int = 5,
                 history_limit: int = 200):
        """
        Args:
            path         : where to write (must match server.py STATE_PATHS)
            write_every  : write to disk every N calls (reduces I/O)
            history_limit: max history points kept in memory
        """
        self.path          = Path(path)
        self.write_every   = write_every
        self.history_limit = history_limit
        self._history      = []
        self._call_count   = 0

        print(f"[LiveState] Writing to '{self.path}' every {write_every} steps")

    # ------------------------------------------------------------------ #
    #  MAIN UPDATE — call once per training step                          #
    # ------------------------------------------------------------------ #
    def update(self, *, step: int, task_id: int, n_tasks: int,
               loss: float, ewc_loss: float, acc: float,
               net, controller,
               new_events: list = None):
        """
        Buffer metrics, flush to disk every `write_every` steps.
        new_events: list of growth/prune events that happened THIS step.
        """
        self._call_count += 1

        arch         = net.get_architecture()
        total_params = arch["total_params"]
        nes_scores   = getattr(controller, "_last_nes_scores", [])

        # Always append to rolling history
        self._history.append({
            "step":   step,
            "loss":   round(loss,    5),
            "ewc":    round(ewc_loss, 5),
            "acc":    round(acc,     4),
            "params": total_params
        })
        if len(self._history) > self.history_limit:
            self._history = self._history[-self.history_limit:]

        if self._call_count % self.write_every != 0:
            return   # skip disk write this step

        layers = self._build_layers(net, nes_scores)

        state = {
            "current": {
                "step":      step,
                "task_id":   task_id,
                "n_tasks":   n_tasks,
                "loss":      round(loss,    5),
                "ewc_loss":  round(ewc_loss, 5),
                "acc":       round(acc,     4),
                "running":   True,
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            "network": {
                "input_size":  net.input_size,
                "output_size": net.output_size,
                "total_params": total_params,
                "layers":      layers
            },
            "history":       self._history,
            "recent_events": new_events or []
        }
        self._atomic_write(state)

    # ------------------------------------------------------------------ #
    #  FINISH — mark running=False when training ends                     #
    # ------------------------------------------------------------------ #
    def finish(self, *, step: int, task_id: int, n_tasks: int,
               loss: float, ewc_loss: float, acc: float,
               net, controller):
        """Call once at end of training. Sets running=False in the UI."""
        nes_scores = getattr(controller, "_last_nes_scores", [])
        layers     = self._build_layers(net, nes_scores)
        arch       = net.get_architecture()

        state = {
            "current": {
                "step":      step,
                "task_id":   task_id,
                "n_tasks":   n_tasks,
                "loss":      round(loss,    5),
                "ewc_loss":  round(ewc_loss, 5),
                "acc":       round(acc,     4),
                "running":   False,           # ← triggers "TRAINING COMPLETE" overlay
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            "network": {
                "input_size":  net.input_size,
                "output_size": net.output_size,
                "total_params": arch["total_params"],
                "layers":      layers
            },
            "history":       self._history,
            "recent_events": []
        }
        self._atomic_write(state)
        print(f"[LiveState] Final state written — training complete overlay will show")

    # ------------------------------------------------------------------ #
    #  HELPERS                                                            #
    # ------------------------------------------------------------------ #
    def _build_layers(self, net, nes_scores: list) -> list:
        """Build the layers array brain.html needs."""
        layers = []

        # Input node (no NES — it's just data)
        layers.append({
            "id":   0,
            "type": "input",
            "size": net.input_size,
            "nes":  0.0
        })

        # Hidden layers — include live NES scores
        for i, layer in enumerate(net.hidden_layers):
            nes = float(nes_scores[i]) if nes_scores and i < len(nes_scores) else 0.0
            layers.append({
                "id":   i + 1,
                "type": "hidden",
                "size": layer.out_features,
                "nes":  round(nes, 4)
            })

        # Output node
        layers.append({
            "id":   len(net.hidden_layers) + 1,
            "type": "output",
            "size": net.output_size,
            "nes":  0.0
        })

        return layers

    def _atomic_write(self, state: dict):
        """
        Write via temp file + rename — prevents brain.html from reading
        a half-written file if training and server run simultaneously.
        """
        tmp = self.path.with_suffix(".tmp")
        try:
            with open(tmp, "w") as f:
                json.dump(state, f, separators=(",", ":"))
            os.replace(tmp, self.path)
        except Exception as e:
            print(f"[LiveState] Write failed: {e}")
