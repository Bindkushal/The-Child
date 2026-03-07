"""
train.py
========
Master training loop — ties all SENN components together.
Run: python train.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import json
import os

from dynamic_net          import SelfExpandingNet
from growth_controller    import GrowthController
from memory_manager       import EWCMemoryManager
from github_self_modifier import GitHubSelfModifier
from live_state_writer    import LiveStateWriter        # ← NEW


# ─────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────
CONFIG = {
    "input_size":          20,
    "output_size":         5,
    "initial_hidden":      [16],
    "lr":                  0.001,
    "batch_size":          32,
    "steps_per_task":      300,
    "n_tasks":             3,
    "ewc_lambda":          400.0,
    "expansion_threshold": 0.15,
    "repo_path":           ".",
    "auto_push":           False,
    "log_every":           50,
}


# ─────────────────────────────────────────────────────
#  SYNTHETIC DATA
# ─────────────────────────────────────────────────────
def make_task_data(task_id, n_samples=1000, input_size=20, output_size=5):
    torch.manual_seed(task_id * 42)
    complexity = 1.0 + task_id * 0.5
    X      = torch.randn(n_samples, input_size)
    W      = torch.randn(input_size, output_size) * complexity
    logits = X @ W + torch.sin(X @ W) * 0.5 * task_id
    y      = logits.argmax(dim=1)
    return DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)


# ─────────────────────────────────────────────────────
#  MAIN TRAINING LOOP
# ─────────────────────────────────────────────────────
def train():
    print("=" * 60)
    print("  SELF-EXPANDING NEURAL NETWORK — Training Start")
    print("=" * 60)

    net = SelfExpandingNet(
        input_size     = CONFIG["input_size"],
        output_size    = CONFIG["output_size"],
        initial_hidden = CONFIG["initial_hidden"]
    )
    print(f"\nInitial network: {net}\n")

    optimizer  = optim.Adam(net.parameters(), lr=CONFIG["lr"])
    criterion  = nn.CrossEntropyLoss()

    controller = GrowthController(
        net                  = net,
        expansion_threshold  = CONFIG["expansion_threshold"],
        loss_plateau_patience= 50,
        prune_threshold      = 0.005,
        neurons_to_add       = 4
    )

    memory   = EWCMemoryManager(net, ewc_lambda=CONFIG["ewc_lambda"])

    modifier = GitHubSelfModifier(
        repo_path = CONFIG["repo_path"],
        auto_push = CONFIG["auto_push"]
    )

    writer = LiveStateWriter(                          # ← NEW
        path        = "live_state.json",
        write_every = 5,
    )

    global_step  = 0
    training_log = []
    last_acc     = 0.0
    last_loss    = 1.0

    # ─────────────────────────────────────────────────
    #  MULTI-TASK LOOP
    # ─────────────────────────────────────────────────
    for task_id in range(CONFIG["n_tasks"]):
        print(f"\n{'─'*60}")
        print(f"  TASK {task_id + 1}/{CONFIG['n_tasks']}")
        print(f"{'─'*60}")

        dataloader   = make_task_data(task_id,
                                      input_size  = CONFIG["input_size"],
                                      output_size = CONFIG["output_size"])
        net.train()
        step_in_task = 0

        for epoch in range(CONFIG["steps_per_task"] // len(dataloader) + 1):
            for x_batch, y_batch in dataloader:
                if step_in_task >= CONFIG["steps_per_task"]:
                    break

                # ── Forward ──
                optimizer.zero_grad()
                outputs     = net(x_batch)
                task_loss   = criterion(outputs, y_batch)
                ewc_penalty = memory.ewc_loss()
                total_loss  = task_loss + ewc_penalty
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()

                loss_val = task_loss.item()

                # ── Growth check ──
                events_before = len(controller.growth_events)   # ← NEW
                grew = controller.step(
                    loss      = loss_val,
                    optimizer = optimizer,
                    outputs   = outputs.detach()
                )

                # Collect events that fired THIS step                # ← NEW
                new_events = []
                if grew:
                    for e in controller.growth_events[events_before:]:
                        new_events.append({
                            "type":    "grow",
                            "layer":   e.get("layer", 0),
                            "count":   e.get("added", e.get("size", 1)),
                            "step":    global_step,
                            "trigger": e.get("trigger", "auto")
                        })
                    memory.handle_growth()
                    if controller.growth_events:
                        modifier.on_growth_event(
                            net          = net,
                            growth_event = controller.growth_events[-1],
                            loss         = loss_val,
                            step         = global_step
                        )

                # ── Live state update ──                            # ← NEW
                last_acc  = _compute_accuracy(net, dataloader)
                last_loss = loss_val
                writer.update(
                    step       = global_step,
                    task_id    = task_id + 1,
                    n_tasks    = CONFIG["n_tasks"],
                    loss       = loss_val,
                    ewc_loss   = ewc_penalty.item(),
                    acc        = last_acc,
                    net        = net,
                    controller = controller,
                    new_events = new_events if grew else []
                )

                # ── Console log ──
                if global_step % CONFIG["log_every"] == 0:
                    arch      = net.get_architecture()
                    log_entry = {
                        "step":         global_step,
                        "task":         task_id + 1,
                        "loss":         round(loss_val, 5),
                        "ewc_penalty":  round(ewc_penalty.item(), 5),
                        "accuracy":     round(last_acc, 3),
                        "n_params":     arch["total_params"],
                        "architecture": str(net)
                    }
                    training_log.append(log_entry)
                    print(f"  Step {global_step:>5} | Task {task_id+1} | "
                          f"Loss={loss_val:.4f} | EWC={ewc_penalty.item():.4f} | "
                          f"Acc={last_acc:.2%} | Params={arch['total_params']:,}")
                    print(f"           Net: {net}")

                global_step  += 1
                step_in_task += 1

        # ── Consolidate memory after task ──
        print(f"\n[Memory] Consolidating Task {task_id + 1}...")
        memory.after_task(
            dataloader = dataloader,
            criterion  = criterion,
            task_name  = f"task_{task_id + 1}"
        )

    # ─────────────────────────────────────────────────
    #  FINAL REPORT
    # ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nFinal architecture: {net}")
    print(f"\nGrowth report:")
    print(json.dumps(controller.report(), indent=2))
    print(f"\nMemory report:")
    print(json.dumps(memory.report(), indent=2))

    with open("training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)
    print(f"\n✓ Training log saved to training_log.json")

    # Mark visualizer complete                                       # ← NEW
    writer.finish(
        step       = global_step,
        task_id    = CONFIG["n_tasks"],
        n_tasks    = CONFIG["n_tasks"],
        loss       = last_loss,
        ewc_loss   = 0.0,
        acc        = last_acc,
        net        = net,
        controller = controller,
    )

    torch.save(net.state_dict(), "senn_final.pt")
    print("✓ Model saved: senn_final.pt")
    return net


# ─────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────
def _compute_accuracy(net, dataloader, max_batches=5):
    net.eval()
    correct = total = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            if i >= max_batches:
                break
            preds    = net(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += len(y)
    net.train()
    return correct / max(total, 1)


if __name__ == "__main__":
    GitHubSelfModifier.print_setup_instructions(
        repo_path  = ".",
        github_url = "https://github.com/Bindkushal/The-Child"
    )
    train()
