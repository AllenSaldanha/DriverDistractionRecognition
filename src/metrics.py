import os
import argparse
import pandas as pd

class Metric:
    """Metrics for multi-label activity detection in video clips."""

    def __init__(self, ground_truth_csv, prediction_folder):
        self.ground_truth = None
        # self.filtered_ground_truth = self.ground_truth[self.ground_truth['file_id'] == file_id]
        self.predictions = self.parse_predictions(prediction_folder)
        self.segmented_predictions = self.convert_predictions_to_segments(self.predictions)

    @staticmethod
    def parse_predictions(prediction_folder):
        """
        Parse all prediction CSVs in the folder for the given file_id (video stem).
        Returns a DataFrame with columns: frame, activity, file_id.
        """
        records = []
        for pred_file in os.listdir(prediction_folder):
            if pred_file.endswith(".csv"):
                csv_file = os.path.join(prediction_folder, pred_file)
                df = pd.read_csv(csv_file)
                for _, row in df.iterrows():
                    start_frame = int(row['Frame Start'])
                    label_str = row['Predicted Labels']
                    if pd.isna(label_str) or label_str.strip() == "":
                        continue
                    labels = [label.strip() for label in label_str.split(';') if label.strip()]
                    for label in labels:
                        records.append({'frame': start_frame, 'activity': label, 'pred_file': pred_file})
        return pd.DataFrame(records)

    @staticmethod
    def convert_predictions_to_segments(predictions):
        """
        Converts frame-level predictions to activity segments.
        """
        if predictions.empty:
            return pd.DataFrame(columns=['frame_start', 'frame_end', 'activity'])
        # Sort by frame
        predictions = predictions.sort_values(by='frame')
        segments = []
        current_activity = None
        current_start = None

        for _, row in predictions.iterrows():
            frame = row['frame']
            activity = row['activity']

            # Start a new segment if the activity changes
            if activity != current_activity:
                if current_activity is not None:
                    # Save the previous segment
                    segments.append({
                        'frame_start': current_start,
                        'frame_end': frame - 1,
                        'activity': current_activity
                    })
                # Start a new segment
                current_activity = activity
                current_start = frame

        # Save the last segment
        if current_activity is not None:
            segments.append({
                'frame_start': current_start,
                'frame_end': predictions.iloc[-1]['frame'],
                'activity': current_activity
            })

        return pd.DataFrame(segments)
    
    
    def evaluate_multiclass(self):
        all_activities = set(self.filtered_ground_truth['activity']).union(set(self.segmented_predictions['activity']))

        # Initialize metrics for all known activities
        metrics = {cls: {'tp': 0, 'fp': 0, 'fn': 0} for cls in all_activities}
        matched_chunks = set()  
        
        for _, gt in self.filtered_ground_truth.iterrows():
            gt_start = gt['frame_start']
            gt_end = gt['frame_end']
            gt_activity = gt['activity']
            chunk_key = (gt['annotation_id'], gt['chunk_id'])
            gt_midpoint = (gt_start + gt_end) // 2

            matched_prediction = self.segmented_predictions[
                (self.segmented_predictions['frame_start'] <= gt_midpoint) &
                (self.segmented_predictions['frame_end'] >= gt_midpoint) &
                (self.segmented_predictions['activity'] == gt_activity)
            ]

            if not matched_prediction.empty:
                metrics[gt_activity]['tp'] += 1
            else:
                metrics[gt_activity]['fn'] += 1

        # Count False Positives for unmatched predictions
        for _, pred in self.segmented_predictions.iterrows():
            pred_activity = pred['activity']
            pred_midpoint = (pred['frame_start'] + pred['frame_end']) // 2
            matched_gt = self.filtered_ground_truth[
                (self.filtered_ground_truth['frame_start'] <= pred_midpoint) &
                (self.filtered_ground_truth['frame_end'] >= pred_midpoint) &
                (self.filtered_ground_truth['activity'] == pred_activity)
            ]
            if matched_gt.empty:
                metrics[pred_activity]['fp'] += 1

        precision, recall = {}, {}
        for cls in all_activities:
            tp, fp, fn = metrics[cls]['tp'], metrics[cls]['fp'], metrics[cls]['fn']
            precision[cls] = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
            recall[cls] = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
            
        overall_tp = sum(metrics[cls]['tp'] for cls in all_activities)
        overall_fp = sum(metrics[cls]['fp'] for cls in all_activities)
        overall_fn = sum(metrics[cls]['fn'] for cls in all_activities)

        overall_precision = overall_tp / (overall_tp + overall_fp) * 100 if (overall_tp + overall_fp) > 0 else 0
        overall_recall = overall_tp / (overall_tp + overall_fn) * 100 if (overall_tp + overall_fn) > 0 else 0

        return precision, recall, overall_precision, overall_recall

    def iou(self):
        """
        Compute mean IoU between ground truth and predicted activity segments.
        """
        iou_scores = []
        for _, gt in self.filtered_ground_truth.iterrows():
            gt_start = gt['frame_start']
            gt_end = gt['frame_end']
            activity = gt['activity']

            # Find all predicted frames that match the activity
            pred_segments = self.segmented_predictions[
                (self.segmented_predictions['activity'] == activity)
            ]
            if pred_segments.empty:
                iou_scores.append(0)
                continue

            # Compute IoU for each overlap, take max
            best_iou = 0
            for _, pred in pred_segments.iterrows():
                pred_start = pred['frame_start']
                pred_end = pred['frame_end']
                intersection_start = max(gt_start, pred_start)
                intersection_end = min(gt_end, pred_end)
                intersection = max(0, intersection_end - intersection_start + 1)
                union_start = min(gt_start, pred_start)
                union_end = max(gt_end, pred_end)
                union = max(0, union_end - union_start + 1)
                iou = intersection / union if union > 0 else 0
                best_iou = max(best_iou, iou)
            iou_scores.append(best_iou)
        return sum(iou_scores) / len(iou_scores) if iou_scores else 0.0

    def evaluate(self):
        mean_iou = self.iou()
        precision, recall, overall_precision, overall_recall = self.evaluate_multiclass()
        print("Precision per class (%):", {cls: f"{precision[cls]:.2f}%" for cls in precision})
        print("Recall per class (%):", {cls: f"{recall[cls]:.2f}%" for cls in recall})
        print(f"Overall Precision: {overall_precision:.2f}%")
        print(f"Overall Recall: {overall_recall:.2f}%")
        print(f"Mean IoU: {mean_iou*100:.2f}%")
        return {
            "Precision per class (%)": {cls: f"{precision[cls]:.2f}%" for cls in precision},
            "Recall per class (%)": {cls: f"{recall[cls]:.2f}%" for cls in recall},
            "Overall Precision": f"{overall_precision:.2f}%",
            "Overall Recall": f"{overall_recall:.2f}%",
            "Mean IoU": f"{mean_iou*100:.2f}%"
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground_truth", default="./dataset/dmd/gA", help="Path to the ground truth CSV file")
    parser.add_argument("--prediction_log", default="./inference_logs", help="Path to the prediction log folder")
    parser.add_argument("--output_file", default="./metrics", help="Path to save the evaluation results")
    args = parser.parse_args()

    metrics = Metric(args.ground_truth, args.prediction_log)
    results = metrics.evaluate()
    
    with open(args.output_file, 'w') as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    print(f"Metrics evaluation saved to {args.output_file}")