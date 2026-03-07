"""
train.py
========
Master training loop — Self-Expanding Neural Network.

What this does:
  - Task 1: Teaches the network to recognise handwritten letters A-Z
  - Task 2: Teaches handwritten digits 0-9 WITHOUT forgetting letters
  - Network starts with 128 neurons and grows as large as needed
  - Every growth event is committed to GitHub automatically
  - Brain visualizer updates live while training runs

Run locally : python train.py
Run on Colab: use The_AI_Child.ipynb
"""

import torch
import torch.nn as nn
import torch.optim as optim
import json
import os

from dynamic_net          import SelfExpandingNet
from growth_controller    import GrowthController
from memory_manager       import EWCMemoryManager
from github_self_modifier import GitHubSelfModifier
from live_state_writer    import LiveStateWriter
from data_loader          import get_letters_loader, get_digits_loader


# ─────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────
CONFIG = {
    # Network shape
    "input_size":      784,    # 28x28 pixels flattened
    "output_size":     26,     # 26 letters A-Z
    "initial_hidden":  [128],  # ← start bigger so neurons survive early training

    # Training
    "lr":              0.001,
    "batch_size":      64,
    "steps_per_task":  3000,   # ← more steps = more time to learn and grow

    # Continual learning
    "n_tasks":         1,      # letters only for now

    # EWC memory
    "ewc_lambda":      400.0,

    # Growth — how eagerly the network expands
    "expansion_threshold": 0.03,   # ← low = grows often and early

    # GitHub
    "repo_path":  ".",
    "auto_push":  False,       # set True in Colab after git config

    # Logging
    "log_every":  100,
    "save_dir":   "models",
}


# ─────────────────────────────────────────────────────────────────────────
#  TASK DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────
TASKS = [
    {
        "name":        "Letters A-Z",
        "loader_fn":   lambda: get_letters_loader(batch_size=CONFIG["batch_size"]),
        "description": "Learning to read handwritten letters A through Z"
    },
    # Uncomment Task 2 once letters reach >60% accuracy:
    # {
    #     "name":        "Digits 0-9",
    #     "loader_fn":   lambda: get_digits_loader(batch_size=CONFIG["batch_size"]),
    #     "description": "Learning digits while remembering letters (EWC active)"
    # },
]


# ─────────────────────────────────────────────────────────────────────────
#  MAIN TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────────────────
def train():
    print("=" * 60)
    print("  SELF-EXPANDING NEURAL NETWORK")
    print("  The AI Child — Training Session")
    print("=" * 60)

    os.makedirs(CONFIG["save_dir"], exist_ok=True)

    # ── Build the network ─────────────────────────────────────────────────
    net = SelfExpandingNet(
        input_size     = CONFIG["input_size"],
        output_size    = CONFIG["output_size"],
        initial_hidden = CONFIG["initial_hidden"]
    )
    print(f"\nStarting network : {net}")
    print(f"Starting params  : {sum(p.numel() for p in net.parameters()):,}")
    print(f"Tasks to learn   : {[t['name'] for t in TASKS]}\n")

    # ── Tools ─────────────────────────────────────────────────────────────
    optimizer = optim.Adam(net.parameters(), lr=CONFIG["lr"])
    criterion = nn.CrossEntropyLoss()

    controller = GrowthController(
        net                   = net,
        expansion_threshold   = CONFIG["expansion_threshold"],
        loss_plateau_patience = 150,   # wait longer before plateau trigger
        prune_threshold       = 0.0,   # ← pruning OFF — let network grow freely
        neurons_to_add        = 32,    # ← add 32 neurons per growth event (was 8)
        max_neurons_per_layer = 1024,  # ← allow very large layers
    )

    memory = EWCMemoryManager(net, ewc_lambda=CONFIG["ewc_lambda"])

    modifier = GitHubSelfModifier(
        repo_path = CONFIG["repo_path"],
        auto_push = CONFIG["auto_push"]
    )

    writer = LiveStateWriter(
        path        = "live_state.json",
        write_every = 10,
    )

    # ── State ─────────────────────────────────────────────────────────────
    global_step  = 0
    training_log = []
    last_acc     = 0.0
    last_loss    = 1.0

    # ─────────────────────────────────────────────────────────────────────
    #  TASK LOOP
    # ─────────────────────────────────────────────────────────────────────
    for task_id, task in enumerate(TASKS):
        print(f"\n{'='*60}")
        print(f"  TASK {task_id+1}/{len(TASKS)}: {task['name']}")
        print(f"  {task['description']}")
        print(f"{'='*60}")

        print("\nLoading data...")
        train_loader, val_loader = task["loader_fn"]()

        net.train()
        step_in_task = 0
        best_val_acc = 0.0

        # ── STEPS LOOP ────────────────────────────────────────────────────
        while step_in_task < CONFIG["steps_per_task"]:
            for x_batch, y_batch in train_loader:
                if step_in_task >= CONFIG["steps_per_task"]:
                    break

                # Forward
                optimizer.zero_grad()
                outputs     = net(x_batch)
                task_loss   = criterion(outputs, y_batch)
                ewc_penalty = memory.ewc_loss()
                total_loss  = task_loss + ewc_penalty

                # Backward
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()

                loss_val = task_loss.item()

                # Growth check
                events_before = len(controller.growth_events)
                grew = controller.step(
                    loss      = loss_val,
                    optimizer = optimizer,
                    outputs   = outputs.detach()
                )

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

                # Live visualizer
                last_acc  = _quick_accuracy(net, x_batch, y_batch)
                last_loss = loss_val
                writer.update(
                    step       = global_step,
                    task_id    = task_id + 1,
                    n_tasks    = len(TASKS),
                    loss       = loss_val,
                    ewc_loss   = ewc_penalty.item(),
                    acc        = last_acc,
                    net        = net,
                    controller = controller,
                    new_events = new_events if grew else []
                )

                # Console log
                if global_step % CONFIG["log_every"] == 0:
                    val_acc = _compute_accuracy(net, val_loader)
                    arch    = net.get_architecture()
                    log_entry = {
                        "step":        global_step,
                        "task":        task["name"],
                        "loss":        round(loss_val, 5),
                        "ewc_penalty": round(ewc_penalty.item(), 5),
                        "train_acc":   round(last_acc, 3),
                        "val_acc":     round(val_acc, 3),
                        "n_params":    arch["total_params"],
                        "architecture": str(net)
                    }
                    training_log.append(log_entry)

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        _save_checkpoint(net, arch, task["name"],
                                         global_step, val_acc)

                    print(
                        f"  Step {global_step:>5} | "
                        f"Loss={loss_val:.4f} | "
                        f"TrainAcc={last_acc:.1%} | "
                        f"ValAcc={val_acc:.1%} | "
                        f"Params={arch['total_params']:,} | "
                        f"{net}"
                    )

                global_step  += 1
                step_in_task += 1

        # End of task — consolidate memory
        print(f"\n[Memory] Consolidating '{task['name']}'...")
        memory.after_task(
            dataloader = train_loader,
            criterion  = criterion,
            task_name  = task["name"]
        )

        final_val = _compute_accuracy(net, val_loader, max_batches=50)
        print(f"  Final validation accuracy: {final_val:.2%}")

    # ─────────────────────────────────────────────────────────────────────
    #  TRAINING COMPLETE
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)

    arch = net.get_architecture()
    print(f"\nFinal network    : {net}")
    print(f"Total parameters : {arch['total_params']:,}")
    print(f"Growth events    : {arch['growth_events']}")

    print("\nGrowth summary:")
    print(json.dumps(controller.report(), indent=2))

    print("\nMemory summary:")
    print(json.dumps(memory.report(), indent=2))

    # Save weights
    torch.save(net.state_dict(), "senn_final.pt")
    torch.save(net.state_dict(), os.path.join(CONFIG["save_dir"], "senn_final.pt"))
    print(f"\n✓ senn_final.pt saved")

    # Save architecture
    with open("architecture_final.json", "w") as f:
        json.dump(arch, f, indent=2)
    print(f"✓ architecture_final.json saved")

    # Save training log
    log_path = os.path.join(CONFIG["save_dir"], "training_log.json")
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)
    print(f"✓ training_log.json saved")

    # Tell visualizer we are done
    writer.finish(
        step       = global_step,
        task_id    = len(TASKS),
        n_tasks    = len(TASKS),
        loss       = last_loss,
        ewc_loss   = 0.0,
        acc        = last_acc,
        net        = net,
        controller = controller,
    )

    print(f"\n✓ Ready to push to Hugging Face")
    return net


# ─────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────
def _quick_accuracy(net, x_batch, y_batch) -> float:
    net.eval()
    with torch.no_grad():
        preds   = net(x_batch).argmax(dim=1)
        correct = (preds == y_batch).sum().item()
    net.train()
    return correct / len(y_batch)


def _compute_accuracy(net, dataloader, max_batches: int = 10) -> float:
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


def _save_checkpoint(net, arch, task_name, step, val_acc):
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    safe_name = task_name.replace(" ", "_").replace("-", "").lower()
    path = os.path.join(CONFIG["save_dir"], f"best_{safe_name}.pt")
    torch.save({
        "state_dict":   net.state_dict(),
        "architecture": arch,
        "step":         step,
        "val_acc":      val_acc,
    }, path)


# ─────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    GitHubSelfModifier.print_setup_instructions(
        repo_path  = ".",
        github_url = "https://github.com/Bindkushal/The-Child"
    )
    trained_net = train()
