import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt


def iou(boxA, boxB):
    """
    Compute IoU between two bounding boxes in COCO format [x, y, width, height].
    """
    xA1, yA1, wA, hA = boxA
    xB1, yB1, wB, hB = boxB
    xA2, yA2 = xA1 + wA, yA1 + hA
    xB2, yB2 = xB1 + wB, yB1 + hB

    # Intersection
    xI1 = max(xA1, xB1)
    yI1 = max(yA1, yB1)
    xI2 = min(xA2, xB2)
    yI2 = min(yA2, yB2)
    interW = max(0, xI2 - xI1)
    interH = max(0, yI2 - yI1)
    interArea = interW * interH

    # Union
    areaA = wA * hA
    areaB = wB * hB
    unionArea = areaA + areaB - interArea
    if unionArea == 0:
        return 0.0
    return interArea / unionArea


def load_annotations(gt_path, dt_path):
    """
    Load ground truth and detection annotations from COCO JSON files,
    grouping by image_id.
    Returns:
      gt_dict: {image_id: [bbox, ...]}
      dt_dict: {image_id: [(bbox, score), ...]}
    """
    gt_data = json.load(open(gt_path))
    dt_data = json.load(open(dt_path))

    gt_dict = {}
    for ann in gt_data.get('annotations', []):
        gt_dict.setdefault(ann['image_id'], []).append(ann['bbox'])

    dt_dict = {}
    for ann in dt_data:
        dt_dict.setdefault(ann['image_id'], []).append((ann['bbox'], ann.get('score', 1.0)))

    return gt_dict, dt_dict


def compute_ap(recalls, precisions):
    """
    Compute Average Precision (AP) as the area under the precision-recall curve.
    Uses numerical integration (trapezoidal rule).
    """
    # Sort by recall
    idx = np.argsort(recalls)
    recalls = np.array(recalls)[idx]
    precisions = np.array(precisions)[idx]
    # Integrate
    return float(np.trapz(precisions, recalls))


def evaluate_manual(gt_path, dt_path, iou_thresholds=None):
    """
    Manually compute detection metrics:
      - AP at each IoU threshold
      - mAP (mean AP)
      - Precision, Recall, F1 (at IoU=0.5)
    """
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05)

    gt_dict, dt_dict = load_annotations(gt_path, dt_path)
    all_aps = []

    # For each IoU threshold, compute per-image TP/FP/FN
    for thr in iou_thresholds:
        all_recalls = []
        all_precisions = []
        for img_id, gts in gt_dict.items():
            dts = dt_dict.get(img_id, [])
            # sort detections by score desc
            dts = sorted(dts, key=lambda x: x[1], reverse=True)
            matched_gt = set()
            tp = 0
            fp = 0
            for bbox, score in dts:
                # find best matching GT
                best_iou = 0
                best_j = -1
                for j, gt_bbox in enumerate(gts):
                    if j in matched_gt:
                        continue
                    iou_val = iou(bbox, gt_bbox)
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_j = j
                if best_iou >= thr:
                    tp += 1
                    matched_gt.add(best_j)
                else:
                    fp += 1
            fn = len(gts) - len(matched_gt)
            # accumulate precision/recall for this image
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            all_precisions.append(precision)
            all_recalls.append(recall)
        # AP as average of per-image precision-recall
        ap = compute_ap(all_recalls, all_precisions)
        all_aps.append(ap)

    mAP = float(np.mean(all_aps))

    # Use IoU=0.5 metrics for precision/recall/F1
    pr_index = 0  # threshold 0.5 at index 0
    precision_05 = all_precisions if False else None  # placeholder
    recall_05 = all_recalls if False else None  # placeholder
    # recompute global counts at IoU=0.5 to get overall TP/FP/FN
    thr = iou_thresholds[0]
    total_tp = total_fp = total_fn = 0
    for img_id, gts in gt_dict.items():
        dts = dt_dict.get(img_id, [])
        dts = sorted(dts, key=lambda x: x[1], reverse=True)
        matched_gt = set()
        for bbox, score in dts:
            best_iou = 0
            best_j = -1
            for j, gt_bbox in enumerate(gts):
                if j in matched_gt:
                    continue
                iou_val = iou(bbox, gt_bbox)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_j = j
            if best_iou >= thr:
                total_tp += 1
                matched_gt.add(best_j)
            else:
                total_fp += 1
        total_fn += len(gts) - len(matched_gt)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics = {
        'mAP': mAP,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    }
    return metrics


def plot_metrics(metrics, save_path='metrics_bar.png'):
    names = list(metrics.keys())
    values = [metrics[k] for k in names]

    plt.figure(figsize=(8, 5))
    plt.bar(names, values)
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.title('EfficientDet-D0 Metrics')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Bar chart saved to '{save_path}'.")


def save_metrics(metrics, save_path='metrics.json'):
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to '{save_path}'.")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate predictions manually.')
    parser.add_argument('--gt', default='coco_eval/instances_val_gt.json', help='Ground truth JSON')
    parser.add_argument('--preds', default='coco_eval/instances_val_preds.json', help='Predictions JSON')
    parser.add_argument('--output_dir', default='coco_eval', help='Save outputs here')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    try:
        metrics = evaluate_manual(args.gt, args.preds)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Save and plot
    save_metrics(metrics, os.path.join(args.output_dir, 'metrics.json'))
    plot_metrics(metrics, os.path.join(args.output_dir, 'metrics_bar.png'))

    print("Results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print("Evaluation completed.")