"""
Generalization Gap & Overfitting Diagnostics
============================================
Tracks the empirical generalization gap between training and validation losses
across epochs to test the structural regularization hypothesis of register tokens.

Mathematical Formulation:
    Delta_L(epoch) = L_val(epoch) - L_train(epoch)
    Delta_Acc(epoch) = Acc_train(epoch) - Acc_val(epoch)

A smaller, stable generalization gap indicates effective regularization under
data scarcity, whereas a diverging gap indicates unconstrained overfitting.

Public Interface:
- `compute_generalization_gap`: Computes L_val - L_train.
- `compute_accuracy_gap`: Computes Acc_train - Acc_val.
- `GeneralizationGapTracker`: State tracker recording loss trajectories and gaps.
"""

from typing import Dict, List


def compute_generalization_gap(train_loss: float, val_loss: float) -> float:
    """
    Computes single-epoch generalization loss gap: Delta_L = L_val - L_train.

    :param train_loss: Scalar cross-entropy loss on the training split.
    :param val_loss: Scalar cross-entropy loss on the validation split.
    :return: Float difference (val_loss - train_loss).
    """
    return float(val_loss - train_loss)


def compute_accuracy_gap(train_acc: float, val_acc: float) -> float:
    """
    Computes single-epoch generalization accuracy gap: Delta_Acc = Acc_train - Acc_val.

    :param train_acc: Top-1 accuracy percentage on training split (e.g. 85.4).
    :param val_acc: Top-1 accuracy percentage on validation split (e.g. 72.1).
    :return: Float difference in accuracy percentage points.
    """
    return float(train_acc - val_acc)


class GeneralizationGapTracker:
    """
    Tracks and aggregates (epoch, train_loss, val_loss, gap) records across
    an entire training run for visualization, CSV export, and reporting.
    """

    def __init__(self):
        """Initializes empty history record list."""
        self.history: List[Dict[str, float]] = []

    def update(self, epoch: int, train_loss: float, val_loss: float) -> float:
        """
        Records losses for a given epoch and computes the corresponding gap.

        :param epoch: Current training epoch index.
        :param train_loss: Epoch average training loss.
        :param val_loss: Epoch average validation loss.
        :return: Computed generalization gap for this epoch.
        """
        gap = compute_generalization_gap(train_loss, val_loss)
        self.history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "gap": float(gap),
            }
        )
        return gap

    def as_dict_of_lists(self) -> Dict[str, List[float]]:
        """
        Exports history as column-oriented lists for direct integration with
        pandas or matplotlib plotting functions.

        :return: Dict mapping field names to lists of values.
        """
        if not self.history:
            return {"epoch": [], "train_loss": [], "val_loss": [], "gap": []}
        return {
            key: [record[key] for record in self.history]
            for key in self.history[0]
        }

    def final_gap(self) -> float:
        """
        Returns the generalization gap at the latest recorded epoch.

        :return: Final epoch generalization gap.
        :raises ValueError: If tracker history is empty.
        """
        if not self.history:
            raise ValueError("Cannot retrieve final_gap: No epochs recorded in tracker history.")
        return self.history[-1]["gap"]
