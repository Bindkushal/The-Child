"""
growth_controller.py
====================
Decides WHEN / WHERE / HOW to expand the network.

Based on:
  - "Self-Expanding Neural Networks" Mitchell et al. 2023 (arXiv:2307.04526)
    → Natural Expansion Score:  NES = g^T * F^{-1} * g
      where g = gradient vector, F = Fisher Information Matrix
    → Expand when NES > threshold τ

  - "DeDNN" (ScienceDirect 2023)
    → Add NEURONS when local error is high
    → Add LAYERS  when Jensen-Shannon divergence of activations changes sharply

  - "DEN" Yoon et al. ICLR 2018
    → Expand only when loss does NOT drop below threshold after selective retrain
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional
from dynamic_net import SelfExpandingNet, DynamicLayer


class GrowthController:
    """
    The brain that decides when and where the network should grow.

    Three triggers (from literature):
    ──────────────────────────────────
    1. NATURAL EXPANSION SCORE (NES)  ← main trigger (Mitchell et al.)
       Score = gradient magnitude × inverse Fisher approx
       Simple approx: NES ≈ mean(|grad|²) / (mean(|weight|²) + ε)
       → High NES = network is STRUGGLING here → add neurons

    2. LOSS PLATEAU  ← secondary trigger (DEN paper)
       If loss hasn't improved for N steps → add capacity

    3. CONFIDENCE  ← your idea, inspired by developmental AI
       Low output confidence (high entropy on softmax) → network uncertain → grow
    """

    def __init__(self,
                 net: SelfExpandingNet,
                 expansion_threshold: float = 0.15,    # τ — NES threshold
                 loss_plateau_patience: int = 50,       # steps before plateau trigger
                 prune_threshold: float = 0.005,        # min activation to keep neuron
                 max_neurons_per_layer: int = 256,      # cap on growth
                 neurons_to_add: int = 4):              # how many neurons per event

        self.net                  = net
        self.τ                    = expansion_threshold
        self.plateau_patience     = loss_plateau_patience
        self.prune_threshold      = prune_threshold
        self.max_neurons          = max_neurons_per_layer
        self.neurons_to_add       = neurons_to_add

        # History tracking
        self.loss_history: List[float] = []
        self.nes_history:  List[float] = []
        self.growth_events: List[dict] = []
        self._plateau_counter = 0
        self._best_loss       = float('inf')

        print(f"[GrowthController] τ={self.τ} | patience={self.plateau_patience} | "
              f"prune_thresh={self.prune_threshold} | max_neurons={self.max_neurons}")

    # ------------------------------------------------------------------ #
    #  STEP — call this after every training step                         #
    # ------------------------------------------------------------------ #
    def step(self, loss: float, optimizer: torch.optim.Optimizer,
             outputs: Optional[torch.Tensor] = None) -> bool:
        """
        Check all growth conditions. Expand if triggered.
        Returns True if the network grew.
        """
        self.loss_history.append(loss)

        # --- Compute Natural Expansion Score per layer ---
        nes_scores = self._compute_nes()
        self.nes_history.append(max(nes_scores) if nes_scores else 0.0)

        grew = False

        # TRIGGER 1: NES exceeds threshold
        if nes_scores:
            max_nes     = max(nes_scores)
            max_layer   = nes_scores.index(max_nes)

            if max_nes > self.τ:
                grew = self._grow_width(max_layer, optimizer)

        # TRIGGER 2: Loss plateau
        if not grew:
            grew = self._check_plateau(loss, optimizer)

        # TRIGGER 3: Confidence (for classification tasks)
        if not grew and outputs is not None:
            grew = self._check_confidence(outputs, optimizer)

        # PRUNING — always check (keep network lean)
        self._check_pruning(optimizer)

        return grew

    # ------------------------------------------------------------------ #
    #  NATURAL EXPANSION SCORE  (simplified, computationally cheap)       #
    # ------------------------------------------------------------------ #
    def _compute_nes(self) -> List[float]:
        """
        Compute NES per layer.
        Full NES = g^T F^{-1} g is expensive (needs full Fisher).
        
        CHEAP APPROXIMATION used here (from Mitchell et al. section 4.2):
          NES_layer ≈ ||g||² / (||w||² + ε)
          
        This is the ratio of gradient energy to weight energy.
        High ratio = gradients are large relative to weights = network struggling.
        """
        scores = []
        for i, layer in enumerate(self.net.hidden_layers):
            grad = layer.linear.weight.grad
            weight = layer.linear.weight.data

            if grad is None:
                scores.append(0.0)
                continue

            g_norm  = grad.pow(2).mean().item()
            w_norm  = weight.pow(2).mean().item()
            nes     = g_norm / (w_norm + 1e-8)
            scores.append(nes)

        return scores

    # ------------------------------------------------------------------ #
    #  WIDTH GROWTH — add neurons to layer i                              #
    # ------------------------------------------------------------------ #
    def _grow_width(self, layer_idx: int, optimizer: torch.optim.Optimizer) -> bool:
        layer = self.net.hidden_layers[layer_idx]

        if layer.out_features >= self.max_neurons:
            print(f"  [SKIP]  Layer {layer_idx} already at max ({self.max_neurons})")
            return False

        next_layer = (self.net.hidden_layers[layer_idx + 1]
                      if layer_idx + 1 < len(self.net.hidden_layers)
                      else None)

        # Add neurons
        layer.add_neurons(self.neurons_to_add, next_layer=next_layer)

        # Fix output layer if this was the last hidden layer
        self.net._fix_output_layer()

        # Rebuild optimizer with new parameters
        self._rebuild_optimizer(optimizer)

        event = {
            "type":       "width_growth",
            "layer":      layer_idx,
            "added":      self.neurons_to_add,
            "new_size":   layer.out_features,
            "loss":       self.loss_history[-1] if self.loss_history else 0,
            "trigger":    "NES"
        }
        self.growth_events.append(event)
        return True

    # ------------------------------------------------------------------ #
    #  PLATEAU TRIGGER                                                    #
    # ------------------------------------------------------------------ #
    def _check_plateau(self, loss: float,
                       optimizer: torch.optim.Optimizer) -> bool:
        """
        From DEN paper: if loss doesn't drop → add capacity.
        """
        if loss < self._best_loss * 0.999:   # improved by 0.1%
            self._best_loss       = loss
            self._plateau_counter = 0
            return False

        self._plateau_counter += 1

        if self._plateau_counter >= self.plateau_patience:
            self._plateau_counter = 0
            print(f"  [PLATEAU] No improvement for {self.plateau_patience} steps → growing")

            # Add to the SMALLEST layer (most capacity constrained)
            sizes   = [l.out_features for l in self.net.hidden_layers]
            min_idx = sizes.index(min(sizes))
            return self._grow_width(min_idx, optimizer)

        return False

    # ------------------------------------------------------------------ #
    #  CONFIDENCE TRIGGER  (your idea — "child uncertain → learn more")   #
    # ------------------------------------------------------------------ #
    def _check_confidence(self, outputs: torch.Tensor,
                          optimizer: torch.optim.Optimizer) -> bool:
        """
        If model output entropy is HIGH → model is uncertain → grow.
        Entropy = -Σ p * log(p) ; max entropy = log(num_classes)
        Trigger if mean entropy > 80% of max.
        """
        with torch.no_grad():
            probs   = torch.softmax(outputs, dim=-1)
            entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1).mean().item()
            n_cls   = outputs.shape[-1]
            max_ent = np.log(n_cls) if n_cls > 1 else 1.0
            ratio   = entropy / (max_ent + 1e-8)

        if ratio > 0.80:   # 80% of max entropy = very uncertain
            print(f"  [CONFIDENCE] Entropy ratio={ratio:.3f} > 0.80 → growing")
            # Add a completely NEW layer for better understanding
            if len(self.net.hidden_layers) < 6:   # max depth limit
                self.net.add_layer(size=8)
                self._rebuild_optimizer(optimizer)
                event = {"type": "depth_growth", "trigger": "confidence",
                         "entropy_ratio": ratio}
                self.growth_events.append(event)
                return True
        return False

    # ------------------------------------------------------------------ #
    #  PRUNING                                                            #
    # ------------------------------------------------------------------ #
    def _check_pruning(self, optimizer: torch.optim.Optimizer):
        """
        Prune dead neurons. Run every step (cheap check).
        From DEN paper: group-sparsity removes unnecessary neurons.
        """
        pruned_any = False
        for i, layer in enumerate(self.net.hidden_layers):
            next_l = (self.net.hidden_layers[i + 1]
                      if i + 1 < len(self.net.hidden_layers)
                      else None)
            pruned = layer.prune_neurons(threshold=self.prune_threshold,
                                         next_layer=next_l)
            if pruned > 0:
                pruned_any = True
                self.net._fix_output_layer()

        if pruned_any:
            self._rebuild_optimizer(optimizer)

    # ------------------------------------------------------------------ #
    #  REBUILD OPTIMIZER after architecture change                        #
    # ------------------------------------------------------------------ #
    def _rebuild_optimizer(self, optimizer: torch.optim.Optimizer):
        """
        Architecture changed → old optimizer references stale tensors.
        Rebuild with new parameters, carrying over learning rate.
        """
        lr = optimizer.param_groups[0]['lr']
        optimizer.param_groups.clear()
        optimizer.state.clear()
        new_params = list(self.net.parameters())
        optimizer.add_param_group({'params': new_params, 'lr': lr})

    # ------------------------------------------------------------------ #
    #  REPORT                                                             #
    # ------------------------------------------------------------------ #
    def report(self) -> dict:
        return {
            "total_growth_events": len(self.growth_events),
            "current_architecture": self.net.get_architecture(),
            "growth_log": self.growth_events[-10:]   # last 10 events
        }
