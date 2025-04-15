# metrics.py
import numpy as np
import torch
from monai.metrics import DiceMetric
from monai.metrics.utils import do_metric_reduction, get_mask_edges, get_surface_distance
from monai.metrics import CumulativeIterationMetric

# Custom Hausdorff Score adapted for Kaggle metric (0-1 range, lower is better distance -> higher is better score)
class HausdorffScore(CumulativeIterationMetric):
    """
    Computes the Hausdorff Score (1 - normalized HD).
    A score closer to 1 is better.
    Inherits from CumulativeIterationMetric for epoch-wise aggregation.
    Adapted from uploaded metric.py
    """
    def __init__(self, reduction="mean", percentile=None, directed=False):
        super().__init__()
        self.reduction = reduction
        self.percentile = percentile # Not used in the custom Kaggle version below, but kept for potential MONAI compatibility
        self.directed = directed # If True, computes directed HD, otherwise max(hd(A,B), hd(B,A))

    def _compute_tensor(self, pred: torch.Tensor, gt: torch.Tensor):
        """
        Args:
            pred: Input prediction tensor. Assumed to be binary (0 or 1) after thresholding. Shape (B, C, H, W, D)
            gt: Ground truth tensor. Assumed to be binary (0 or 1). Shape (B, C, H, W, D)
        """
        return _compute_hausdorff_score_batch(y_pred=pred, y=gt, directed=self.directed)

    def aggregate(self):
        """Aggregate the results from all iterations."""
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor):
            raise ValueError("HausdorffScore buffer should be a tensor.")

        # Aggregate based on reduction mode
        f, not_nans = do_metric_reduction(data, self.reduction)
        return f


def _compute_directed_hausdorff(pred_points, gt_points):
    """Computes the directed Hausdorff distance from pred_points to gt_points."""
    if pred_points.shape[0] == 0 or gt_points.shape[0] == 0:
        # Return a large distance if one set is empty and the other isn't
        # Or 0 if both are empty (handled by caller)
        return np.inf if pred_points.shape[0] != gt_points.shape[0] else 0.0

    # Use scipy's cdist for efficient distance calculation
    from scipy.spatial.distance import cdist
    distances = cdist(pred_points, gt_points, metric='euclidean')

    # Distance from each point in pred to nearest in gt
    min_dist_pred_to_gt = distances.min(axis=1)

    # Directed Hausdorff is the max of these minimum distances
    return min_dist_pred_to_gt.max()

def _compute_hausdorff_distance_3d(pred_mask, gt_mask, directed=False):
    """Computes Hausdorff distance between two 3D binary masks."""
    pred_mask = np.asarray(pred_mask, dtype=bool)
    gt_mask = np.asarray(gt_mask, dtype=bool)

    if not np.any(pred_mask) and not np.any(gt_mask):
        return 0.0 # Both empty
    if not np.any(pred_mask) or not np.any(gt_mask):
         return np.inf # One is empty, distance is infinite

    # Get coordinates of non-zero voxels (surface points)
    # Using edges might be more robust but slower; using all points is common.
    pred_coords = np.argwhere(pred_mask)
    gt_coords = np.argwhere(gt_mask)

    if directed:
        # Compute only gt_to_pred distance for Kaggle score ? (Verify this assumption)
        # The Kaggle description isn't explicit on directed vs symmetric. Symmetric is safer.
        # Let's compute symmetric HD for now.
         hd_pred_to_gt = _compute_directed_hausdorff(pred_coords, gt_coords)
         hd_gt_to_pred = _compute_directed_hausdorff(gt_coords, pred_coords)
         return max(hd_pred_to_gt, hd_gt_to_pred)
    else:
         # Compute symmetric Hausdorff Distance
         hd_pred_to_gt = _compute_directed_hausdorff(pred_coords, gt_coords)
         hd_gt_to_pred = _compute_directed_hausdorff(gt_coords, pred_coords)
         return max(hd_pred_to_gt, hd_gt_to_pred)


def _compute_hausdorff_score_batch(y_pred: torch.Tensor, y: torch.Tensor, directed=False):
    """
    Computes Hausdorff score for a batch.
    Score = 1 - (Hausdorff Distance / Max Possible Distance)
    """
    y = y.float().cpu().numpy() > 0.5 # Ensure binary numpy arrays
    y_pred = y_pred.float().cpu().numpy() > 0.5

    batch_size, n_class = y_pred.shape[:2]
    spatial_dims = y_pred.shape[2:] # (H, W, D)
    max_dist = np.sqrt(np.sum([dim**2 for dim in spatial_dims]))

    hd_scores = np.zeros((batch_size, n_class))

    for b in range(batch_size):
        for c in range(n_class):
            pred_mask = y_pred[b, c]
            gt_mask = y[b, c]

            if not np.any(pred_mask) and not np.any(gt_mask):
                # Both empty: Perfect score (HD=0) -> Score=1
                hd_scores[b, c] = 1.0
                continue
            if not np.any(pred_mask) or not np.any(gt_mask):
                 # One empty: Worst score (HD=inf or max_dist) -> Score=0
                 hd_scores[b, c] = 0.0
                 continue

            # Calculate Hausdorff Distance
            hd_dist = _compute_hausdorff_distance_3d(pred_mask, gt_mask, directed=directed)

            # Handle infinite distance (should only happen if one mask is empty, handled above)
            if np.isinf(hd_dist):
                score = 0.0
            else:
                # Normalize and convert to score (1 is best, 0 is worst)
                normalized_hd = min(hd_dist / max_dist, 1.0) # Cap at 1
                score = 1.0 - normalized_hd

            hd_scores[b, c] = score

    return torch.from_numpy(hd_scores).float() # Return as tensor


# Function to get metric instances based on config
def get_metric_functions(config):
    """Creates metric instances based on config."""
    metrics = {}
    metric_names = config.evaluation.get('metrics', ["DiceMetric", "HausdorffScore"])
    dice_params = {
        "include_background": config.evaluation.get('include_background', False),
        "reduction": config.evaluation.get('reduction', 'mean_batch')
    }
    hd_params = {
        "reduction": config.evaluation.get('reduction', 'mean_batch')
        # Add percentile or directed params if needed for MONAI's default HD
    }

    for name in metric_names:
        if name == "DiceMetric":
            metrics[name] = DiceMetric(**dice_params)
        elif name == "HausdorffScore":
            # Use our custom score metric
            metrics[name] = HausdorffScore(**hd_params)
        elif name == "HausdorffDistanceMetric":
             # Use MONAI's default HD metric if preferred (different output range)
             from monai.metrics import HausdorffDistanceMetric
             metrics[name] = HausdorffDistanceMetric(percentile=config.evaluation.get('hausdorff_percentile', None), directed=False, **hd_params)
        else:
            print(f"Warning: Metric '{name}' not recognized. Skipping.")

    return metrics