"""
train.py
========
Master training loop that ties all SENN components together.

Demonstrates:
  - Self-expanding network growing from 16 → ?? neurons
  - EWC memory preventing catastrophic forgetting
  - GitHub self-commits after each growth event
  - Multi-task continual learning

Run: python train.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import json
import os

from dynamic_net         import SelfExpandingNet
from growth_controller   import GrowthController
from memory_manager      import EWCMemoryManager
from github_self_modifier import GitHubSelfModifier


# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────
CONFIG = {
    "input_size":         20,
    "output_size":        5,
    "initial_hidden":     [16],        # Start TINY — grows automatically
    "lr":                 0.001,
    "batch_size":         32,
    "steps_per_task":     300,
    "n_tasks":            3,           # 3 different tasks (continual learning)
    "ewc_lambda":         400.0,
    "expansion_threshold": 0.15,       # NES threshold τ
    "repo_path":          ".",         # Change to your repo path
    "auto_push":          False,       # Set True after git setup
    "log_every":          50,
}


# ─────────────────────────────────────────────────────────────
#  SYNTHETIC DATA (replace with YOUR data)
# ─────────────────────────────────────────────────────────────
def make_task_data(task_id: int, n_samples=1000,
                   input_size=20, output_size=5):
    """
    Generate synthetic classification data for a task.
    Each task has slightly different distribution — simulates new knowledge.
    Replace this with your real data loader.
    """
    torch.manual_seed(task_id * 42)

    # Different pattern per task (harder for each subsequent task)
    complexity = 1.0 + task_id * 0.5
    X = torch.randn(n_samples, input_size)
    # Non-linear target (network must grow to capture this)
    W = torch.randn(input_size, output_size) * complexity
    logits = X @ W + torch.sin(X @ W) * 0.5 * task_id
    y = logits.argmax(dim=1)

    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=32, shuffle=True)


# ─────────────────────────────────────────────────────────────
#  MAIN TRAINING LOOP
# ─────────────────────────────────────────────────────────────
def train():
    print("=" * 60)
    print("  SELF-EXPANDING NEURAL NETWORK — Training Start")
    print("=" * 60)

    # Initialize all components
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

    memory     = EWCMemoryManager(net, ewc_lambda=CONFIG["ewc_lambda"])

    modifier   = GitHubSelfModifier(
        repo_path  = CONFIG["repo_path"],
        auto_push  = CONFIG["auto_push"]
    )

    global_step = 0
    training_log = []

    # ──────────────────────────────────────
    #  MULTI-TASK CONTINUAL LEARNING LOOP
    # ──────────────────────────────────────
    for task_id in range(CONFIG["n_tasks"]):
        print(f"\n{'─'*60}")
        print(f"  TASK {task_id + 1}/{CONFIG['n_tasks']}")
        print(f"{'─'*60}")

        dataloader = make_task_data(task_id, input_size=CONFIG["input_size"],
                                    output_size=CONFIG["output_size"])

        net.train()
        step_in_task = 0

        for epoch in range(CONFIG["steps_per_task"] // len(dataloader) + 1):
            for x_batch, y_batch in dataloader:
                if step_in_task >= CONFIG["steps_per_task"]:
                    break

                # ── Forward pass ──
                optimizer.zero_grad()
                outputs = net(x_batch)

                # ── Task loss + EWC memory loss ──
                task_loss = criterion(outputs, y_batch)
                ewc_penalty = memory.ewc_loss()
                total_loss = task_loss + ewc_penalty

                # ── Backward pass ──
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()

                loss_val = task_loss.item()

                # ── Growth check (every step) ──
                grew = controller.step(
                    loss      = loss_val,
                    optimizer = optimizer,
                    outputs   = outputs.detach()
                )

                if grew:
                    # Tell memory about growth
                    memory.handle_growth()
                    # Self-modify source code + commit to git
                    if controller.growth_events:
                        modifier.on_growth_event(
                            net          = net,
                            growth_event = controller.growth_events[-1],
                            loss         = loss_val,
                            step         = global_step
                        )

                # ── Logging ──
                if global_step % CONFIG["log_every"] == 0:
                    acc = _compute_accuracy(net, dataloader)
                    arch = net.get_architecture()
                    log_entry = {
                        "step":        global_step,
                        "task":        task_id + 1,
                        "loss":        round(loss_val, 5),
                        "ewc_penalty": round(ewc_penalty.item(), 5),
                        "accuracy":    round(acc, 3),
                        "n_params":    arch["total_params"],
                        "architecture": str(net)
                    }
                    training_log.append(log_entry)
                    print(f"  Step {global_step:>5} | Task {task_id+1} | "
                          f"Loss={loss_val:.4f} | EWC={ewc_penalty.item():.4f} | "
                          f"Acc={acc:.2%} | Params={arch['total_params']:,}")
                    print(f"           Net: {net}")

                global_step  += 1
                step_in_task += 1

        # ── After task: consolidate memory ──
        print(f"\n[Memory] Consolidating Task {task_id + 1}...")
        memory.after_task(
            dataloader  = dataloader,
            criterion   = criterion,
            task_name   = f"task_{task_id + 1}"
        )

    # ─────────────────────────────────────
    #  FINAL REPORT
    # ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nFinal architecture: {net}")
    print(f"\nGrowth report:")
    report = controller.report()
    print(json.dumps(report, indent=2))
    print(f"\nMemory report:")
    print(json.dumps(memory.report(), indent=2))

    # Save training log
    with open("training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)
    print(f"\n✓ Training log saved to training_log.json")

    # Save final model
    torch.save(net.state_dict(), "senn_final.pt")
    torch.save(net.get_architecture(), "architecture_final.json")
    print("✓ Model saved: senn_final.pt")

    return net


# ─────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────
def _compute_accuracy(net, dataloader, max_batches=5):
    net.eval()
    correct = total = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            if i >= max_batches:
                break
            preds = net(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += len(y)
    net.train()
    return correct / max(total, 1)


# ─────────────────────────────────────────────────────────────
#  WEB LEARNING (Phase 5 — future)
# ─────────────────────────────────────────────────────────────
def learn_from_web(net, query: str, n_articles: int = 3):
    """
    Phase 5: Autonomous web learning.
    Crawls web for relevant text → tokenizes → trains on it.
    
    PLACEHOLDER — implement after Phase 4 is stable.
    
    Rough plan:
      1. requests + BeautifulSoup to fetch articles
      2. Simple TF-IDF vectorizer to encode text → input_size features
      3. Train with growth controller active
      4. Self-commit after learning
    """
    print(f"[WebLearning] TODO: fetch '{query}', train on {n_articles} articles")
    print("  Install: pip install requests beautifulsoup4 scikit-learn")
    # Implementation in Phase 5


if __name__ == "__main__":
    # Print setup guide for GitHub
    GitHubSelfModifier.print_setup_instructions(
        repo_path  = ".",
        github_url = "https://github.com/YOUR_USERNAME/senn-child-ai"
    )
    
    trained_net = train()
