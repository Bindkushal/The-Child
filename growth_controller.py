"""
growth_controller.py
====================
Decides WHEN / WHERE / HOW to expand the network.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional
from dynamic_net import SelfExpandingNet, DynamicLayer


class GrowthController:
    def __init__(self,
                 net: SelfExpandingNet,
                 expansion_threshold: float = 0.15,
                 loss_plateau_patience: int = 50,
                 prune_threshold: float = 0.005,
                 max_neurons_per_layer: int = 256,
                 neurons_to_add: int = 4):

        self.net              = net
        self.τ                = expansion_threshold
        self.plateau_patience = loss_plateau_patience
        self.prune_threshold  = prune_threshold
        self.max_neurons      = max_neurons_per_layer
        self.neurons_to_add   = neurons_to_add

        self.loss_history: List[float] = []
        self.nes_history:  List[float] = []
        self.growth_events: List[dict] = []
        self._plateau_counter = 0
        self._best_loss       = float('inf')
        self._last_nes_scores: List[float] = []   # ← NEW

        print(f"[GrowthController] τ={self.τ} | patience={self.plateau_patience} | "
              f"prune_thresh={self.prune_threshold} | max_neurons={self.max_neurons}")

    def step(self, loss: float, optimizer: torch.optim.Optimizer,
             outputs=None) -> bool:
        self.loss_history.append(loss)

        nes_scores = self._compute_nes()
        self._last_nes_scores = nes_scores          # ← NEW
        self.nes_history.append(max(nes_scores) if nes_scores else 0.0)

        grew = False

        if nes_scores:
            max_nes   = max(nes_scores)
            max_layer = nes_scores.index(max_nes)
            if max_nes > self.τ:
                grew = self._grow_width(max_layer, optimizer)

        if not grew:
            grew = self._check_plateau(loss, optimizer)

        if not grew and outputs is not None:
            grew = self._check_confidence(outputs, optimizer)

        self._check_pruning(optimizer)
        return grew

    def _compute_nes(self) -> List[float]:
        scores = []
        for i, layer in enumerate(self.net.hidden_layers):
            grad   = layer.linear.weight.grad
            weight = layer.linear.weight.data
            if grad is None:
                scores.append(0.0)
                continue
            g_norm = grad.pow(2).mean().item()
            w_norm = weight.pow(2).mean().item()
            scores.append(g_norm / (w_norm + 1e-8))
        return scores

    def _grow_width(self, layer_idx: int,
                    optimizer: torch.optim.Optimizer) -> bool:
        layer = self.net.hidden_layers[layer_idx]
        if layer.out_features >= self.max_neurons:
            print(f"  [SKIP]  Layer {layer_idx} already at max ({self.max_neurons})")
            return False

        next_layer = (self.net.hidden_layers[layer_idx + 1]
                      if layer_idx + 1 < len(self.net.hidden_layers)
                      else None)

        layer.add_neurons(self.neurons_to_add, next_layer=next_layer)
        self.net._fix_output_layer()
        self._rebuild_optimizer(optimizer)

        self.growth_events.append({
            "type":     "width_growth",
            "layer":    layer_idx,
            "added":    self.neurons_to_add,
            "new_size": layer.out_features,
            "loss":     self.loss_history[-1] if self.loss_history else 0,
            "trigger":  "NES"
        })
        return True

    def _check_plateau(self, loss: float,
                       optimizer: torch.optim.Optimizer) -> bool:
        if loss < self._best_loss * 0.999:
            self._best_loss       = loss
            self._plateau_counter = 0
            return False

        self._plateau_counter += 1
        if self._plateau_counter >= self.plateau_patience:
            self._plateau_counter = 0
            print(f"  [PLATEAU] No improvement for {self.plateau_patience} steps → growing")
            sizes   = [l.out_features for l in self.net.hidden_layers]
            min_idx = sizes.index(min(sizes))
            return self._grow_width(min_idx, optimizer)
        return False

    def _check_confidence(self, outputs: torch.Tensor,
                          optimizer: torch.optim.Optimizer) -> bool:
        with torch.no_grad():
            probs   = torch.softmax(outputs, dim=-1)
            entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1).mean().item()
            n_cls   = outputs.shape[-1]
            max_ent = np.log(n_cls) if n_cls > 1 else 1.0
            ratio   = entropy / (max_ent + 1e-8)

        if ratio > 0.80:
            print(f"  [CONFIDENCE] Entropy ratio={ratio:.3f} > 0.80 → growing")
            if len(self.net.hidden_layers) < 6:
                self.net.add_layer(size=8)
                self._rebuild_optimizer(optimizer)
                self.growth_events.append({
                    "type": "depth_growth",
                    "trigger": "confidence",
                    "entropy_ratio": ratio
                })
                return True
        return False

    def _check_pruning(self, optimizer: torch.optim.Optimizer):
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

    def _rebuild_optimizer(self, optimizer: torch.optim.Optimizer):
        lr = optimizer.param_groups[0]['lr']
        optimizer.param_groups.clear()
        optimizer.state.clear()
        optimizer.add_param_group({'params': list(self.net.parameters()), 'lr': lr})

    def report(self) -> dict:
        return {
            "total_growth_events":  len(self.growth_events),
            "current_architecture": self.net.get_architecture(),
            "growth_log":           self.growth_events[-10:]
        }
