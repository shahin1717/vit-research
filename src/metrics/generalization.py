"""
Generalization Gap Tracker
============================
Tracks the core quantity the whole "regularization" hypothesis hinges on:

    Delta_L(epoch) = L_val(epoch) - L_train(epoch)

A smaller, more stable gap across training epochs is the evidence that
register tokens act as a regularizer under data scarcity. A gap that
grows quickly (train loss keeps dropping, val loss stalls/rises) is the
signature of overfitting.
"""

from typing import Dict, List


def compute_generalization_gap(train_loss: float, val_loss: float) -> float:
    """Single-epoch gap: Delta_L = L_val - L_train."""
    return val_loss - train_loss


class GeneralizationGapTracker:
    """
    Accumulates (epoch, train_loss, val_loss, gap) records across an
    entire training run so the whole curve can be logged to disk and
    plotted (gap vs. epoch, for each K in {0, 1, 4, 8}).
    """

    def __init__(self):
        self.history: List[Dict[str, float]] = []

    def update(self, epoch: int, train_loss: float, val_loss: float) -> float:
        """Records one epoch's losses and returns that epoch's gap."""
        gap = compute_generalization_gap(train_loss, val_loss)
        self.history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "gap": gap,
            }
        )
        return gap

    def as_dict_of_lists(self) -> Dict[str, List[float]]:
        """Convenience export for plotting: {'epoch': [...], 'gap': [...], ...}."""
        if not self.history:
            return {"epoch": [], "train_loss": [], "val_loss": [], "gap": []}
        return {
            key: [record[key] for record in self.history]
            for key in self.history[0]
        }

    def final_gap(self) -> float:
        """Gap at the last recorded epoch -- the headline number for the report."""
        if not self.history:
            raise ValueError("No epochs recorded yet.")
        return self.history[-1]["gap"]
