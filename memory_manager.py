"""
memory_manager.py
=================
Prevents catastrophic forgetting using Elastic Weight Consolidation (EWC).

Based on:
  - Kirkpatrick et al. 2017 "Overcoming catastrophic forgetting in NNs"
  - DeDNN adaptive memory mechanism (ScienceDirect 2023)
  - DEN "semantic drift prevention" (Yoon et al. ICLR 2018)

INTUITION (child learning analogy):
  A child learning maths doesn't forget language.
  EWC puts a "spring" on important weights — they resist changing too much.
  Importance = Fisher Information = how much a weight affects old task performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from typing import Optional, List
from torch.utils.data import DataLoader


class EWCMemoryManager:
    """
    Elastic Weight Consolidation for continual / lifelong learning.

    Usage:
      1. Train on Task A normally
      2. Call memory.after_task(net, dataloader_A)   ← consolidate
      3. Train on Task B with ewc_loss added to your loss
      4. Net remembers Task A while learning Task B
    """

    def __init__(self, net: nn.Module, ewc_lambda: float = 400.0):
        """
        Args:
            net        : the SelfExpandingNet
            ewc_lambda : strength of the memory constraint (higher = more rigid memory)
                         400 is default from Kirkpatrick et al.
        """
        self.net        = net
        self.λ          = ewc_lambda
        self.tasks      = []          # list of (param_means, fisher_diags) per task
        self._task_count = 0

        print(f"[EWCMemory] Initialized with λ={ewc_lambda}")

    # ------------------------------------------------------------------ #
    #  CONSOLIDATE — call AFTER finishing training on a task              #
    # ------------------------------------------------------------------ #
    def after_task(self, dataloader: DataLoader,
                   criterion: nn.Module,
                   n_samples: int = 200,
                   task_name: str = None):
        """
        Compute Fisher Information for current parameters.
        Store as importance weights.

        Fisher diagonal approximation (fast):
          F_i ≈ E[ (∂log p(y|x) / ∂θ_i)² ]
        """
        task_name = task_name or f"task_{self._task_count}"
        print(f"\n[EWCMemory] Consolidating memory for: {task_name}")

        # Save current parameters as "optimal" for this task
        param_means = {}
        for name, param in self.net.named_parameters():
            param_means[name] = param.data.clone()

        # Compute Fisher diagonal
        fisher_diags = {name: torch.zeros_like(param)
                        for name, param in self.net.named_parameters()}

        self.net.eval()
        count = 0

        for x, y in dataloader:
            if count >= n_samples:
                break

            self.net.zero_grad()
            output = self.net(x)

            # Use log-likelihood as objective for Fisher computation
            if output.shape[-1] > 1:
                log_prob = F.log_softmax(output, dim=1)
                # Sample from model's own distribution (Monte Carlo Fisher)
                sampled_y = torch.distributions.Categorical(logits=output).sample()
                loss = F.nll_loss(log_prob, sampled_y)
            else:
                loss = criterion(output.squeeze(), y.float())

            loss.backward()

            for name, param in self.net.named_parameters():
                if param.grad is not None:
                    fisher_diags[name] += param.grad.data.pow(2)

            count += x.shape[0]

        # Normalize
        for name in fisher_diags:
            fisher_diags[name] /= max(count, 1)

        self.tasks.append({
            "name":         task_name,
            "param_means":  param_means,
            "fisher_diags": fisher_diags
        })
        self._task_count += 1
        self.net.train()

        total_params = sum(f.numel() for f in fisher_diags.values())
        print(f"  ✓ Consolidated {total_params:,} params for '{task_name}'")
        print(f"  ✓ Total tasks in memory: {len(self.tasks)}")

    # ------------------------------------------------------------------ #
    #  EWC LOSS — add this to your training loss                          #
    # ------------------------------------------------------------------ #
    def ewc_loss(self) -> torch.Tensor:
        """
        Penalty term to add to task loss:
          L_ewc = λ/2 * Σ_tasks Σ_params F_i * (θ_i - θ_i*)²

        This penalizes moving important weights (high Fisher) far from 
        their optimal values from previous tasks.
        """
        if not self.tasks:
            return torch.tensor(0.0)

        loss = torch.tensor(0.0)

        current_params = {name: param
                          for name, param in self.net.named_parameters()}

        for task in self.tasks:
            for name, param in current_params.items():
                if name not in task["param_means"]:
                    continue   # new params added by growth — no constraint

                mean   = task["param_means"][name]
                fisher = task["fisher_diags"][name]

                # Handle size mismatch (network grew since this task)
                min_shape = tuple(min(s1, s2) for s1, s2 in
                                  zip(param.shape, mean.shape))

                p_slice = param[tuple(slice(0, s) for s in min_shape)]
                m_slice = mean[tuple(slice(0, s) for s in min_shape)]
                f_slice = fisher[tuple(slice(0, s) for s in min_shape)]

                loss += (f_slice * (p_slice - m_slice).pow(2)).sum()

        return (self.λ / 2) * loss

    # ------------------------------------------------------------------ #
    #  UPDATE references after network growth                             #
    # ------------------------------------------------------------------ #
    def handle_growth(self):
        """
        After network grows, new parameters have no Fisher constraint.
        Old parameters retain their constraints (handles size mismatch above).
        This is the "no interference" principle from Mitchell et al.
        """
        # Nothing to do — ewc_loss() handles mismatched sizes gracefully
        print("[EWCMemory] Network grew — old constraints preserved, new params free")

    # ------------------------------------------------------------------ #
    #  MEMORY REPORT                                                      #
    # ------------------------------------------------------------------ #
    def report(self) -> dict:
        return {
            "tasks_remembered": self._task_count,
            "task_names": [t["name"] for t in self.tasks],
            "ewc_lambda": self.λ
        }
