import os
import argparse
import json
import pandas as pd
from pathlib import Path
from dataset import load_trained_classes
from utils.video_annotation_pairs import collect_video_annotation_pairs

class Metric:
    def __init__(self, prediction_folder, pairs):
        self.pairs = pairs
        self.predictions = self.parse_predictions(prediction_folder)
        self.ground_truth_intervals = self.load_ground_truth(pairs)
        
        # Create mapping from prediction file to video path
        self.pred_file_to_video = self.create_pred_file_mapping()
        
        # Get all frames where predictions exist, per video
        self.video_frames = self.get_video_frames()
        
        # Convert ground truth intervals to frame labels per video
        self.gt_frame_labels = self.gt_intervals_to_frame_labels()

    def create_pred_file_mapping(self):
        """Create mapping from prediction CSV file to video path"""
        mapping = {}
        for pred_file in self.predictions['pred_file'].unique():
            # Extract video identifier from prediction file name
            # Adjust this logic based on your naming convention
            pred_base = pred_file.replace('.csv', '')
            
            # Find matching video path
            for video_path, ann_path in self.pairs:
                video_base = Path(video_path).stem
                if pred_base in video_base or video_base in pred_base:
                    mapping[pred_file] = video_path
                    break
            
            if pred_file not in mapping:
                print(f"Warning: Could not find matching video for prediction file {pred_file}")
        
        return mapping

    @staticmethod
    def parse_predictions(prediction_folder):
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
                    labels = set(label.strip() for label in label_str.split(';') if label.strip())
                    records.append({'frame': start_frame, 'labels': labels, 'pred_file': pred_file})
        return pd.DataFrame(records)

    def load_ground_truth(self, pairs):
        gt_records = []
        for video_path, ann_path in pairs:
            with open(ann_path, 'r') as f:
                try:
                    actions = json.load(f)["openlabel"].get("actions", {})
                except Exception as e:
                    print(f"Failed to load {ann_path}: {e}")
                    continue
            for _, info in actions.items():
                activity = info.get("type")
                for interval in info.get("frame_intervals", []):
                    gt_records.append({
                        "frame_start": interval["frame_start"],
                        "frame_end": interval["frame_end"],
                        "activity": activity,
                        "video_path": video_path
                    })
        return pd.DataFrame(gt_records)

    def get_video_frames(self):
        """Get frames per video where predictions exist - with interpolation"""
        video_frames = {}
        for _, row in self.predictions.iterrows():
            pred_file = row['pred_file']
            frame = row['frame']
            video_path = self.pred_file_to_video.get(pred_file)
            if video_path:
                if video_path not in video_frames:
                    video_frames[video_path] = []
                video_frames[video_path].append(frame)
        
        # Sort frames for each video and optionally interpolate
        for video_path in video_frames:
            video_frames[video_path] = sorted(video_frames[video_path])
            
        return video_frames

    def interpolate_predictions(self, video_path, target_frames):
        """Interpolate predictions to cover target frames"""
        # Get predictions for this video
        pred_files = [k for k, v in self.pred_file_to_video.items() if v == video_path]
        video_preds = self.predictions[self.predictions['pred_file'].isin(pred_files)].copy()
        video_preds = video_preds.sort_values('frame')
        
        interpolated_labels = {}
        
        for target_frame in target_frames:
            # Find the closest prediction frame
            pred_frames = video_preds['frame'].values
            closest_idx = min(range(len(pred_frames)), 
                            key=lambda i: abs(pred_frames[i] - target_frame))
            closest_frame = pred_frames[closest_idx]
            
            # Use prediction if within reasonable distance (e.g., 8 frames = 0.5 seconds at 16fps)
            if abs(target_frame - closest_frame) <= 8:
                closest_pred = video_preds[video_preds['frame'] == closest_frame].iloc[0]
                interpolated_labels[target_frame] = closest_pred['labels']
            else:
                interpolated_labels[target_frame] = set()  # No prediction for distant frames
                
        return interpolated_labels

    def gt_intervals_to_frame_labels(self):
        """Convert ground truth intervals to frame labels, per video"""
        gt_frame_labels = {}
        all_activities = set(load_trained_classes("./src/trained_classes.txt"))
        
        for video_path in self.video_frames.keys():
            # Get ground truth for this specific video
            video_gt = self.ground_truth_intervals[
                self.ground_truth_intervals['video_path'] == video_path
            ]
            
            # Get all GT frames for this video
            gt_frames = set()
            for _, row in video_gt.iterrows():
                gt_frames.update(range(row['frame_start'], row['frame_end'] + 1))
            
            # Interpolate predictions for GT frames
            interpolated_preds = self.interpolate_predictions(video_path, gt_frames)
            
            # Create frame labels
            gt_frame_labels[video_path] = {}
            for frame in gt_frames:
                # Ground truth labels for this frame
                active_labels = set()
                for _, row in video_gt.iterrows():
                    if row['frame_start'] <= frame <= row['frame_end']:
                        activity = row['activity']
                        if activity in all_activities:
                            active_labels.add(activity)
                
                gt_frame_labels[video_path][frame] = {
                    'gt_labels': active_labels,
                    'pred_labels': interpolated_preds.get(frame, set())
                }
                
        return gt_frame_labels

    def evaluate_multiclass(self):
        all_activities = load_trained_classes("./src/trained_classes.txt")
        metrics = {cls: {'tp': 0, 'fp': 0, 'fn': 0} for cls in all_activities}

        for video_path, frame_data in self.gt_frame_labels.items():
            for frame, data in frame_data.items():
                pred_labels = data['pred_labels']
                gt_labels = data['gt_labels']

                for cls in all_activities:
                    pred_has = cls in pred_labels
                    gt_has = cls in gt_labels

                    if pred_has and gt_has:
                        metrics[cls]['tp'] += 1
                    elif pred_has and not gt_has:
                        metrics[cls]['fp'] += 1
                    elif not pred_has and gt_has:
                        metrics[cls]['fn'] += 1

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
        """IoU computed frame-wise per video"""
        iou_scores = []

        for video_path, frame_data in self.gt_frame_labels.items():
            for frame, data in frame_data.items():
                pred_labels = data['pred_labels']
                gt_labels = data['gt_labels']

                intersection = len(pred_labels.intersection(gt_labels))
                union = len(pred_labels.union(gt_labels))
                iou = intersection / union if union > 0 else 1.0
                iou_scores.append(iou)

        return sum(iou_scores) / len(iou_scores) if iou_scores else 0.0

    def evaluate(self):
        precision, recall, overall_precision, overall_recall = self.evaluate_multiclass()
        mean_iou = self.iou()
        
        print("Precision per class (%):", {cls: f"{precision[cls]:.2f}%" for cls in precision})
        print("Recall per class (%):", {cls: f"{recall[cls]:.2f}%" for cls in recall})
        print(f"Overall Precision: {overall_precision:.2f}%")
        print(f"Overall Recall: {overall_recall:.2f}%")
        print(f"Mean IoU: {mean_iou*100:.2f}%")
        
        # Print debug info
        all_activities = set(load_trained_classes("./src/trained_classes.txt"))
        gt_activities = set(self.ground_truth_intervals['activity'].unique())
        extra_activities = gt_activities - all_activities
        
        print(f"\nDebug Info:")
        print(f"Total predictions: {len(self.predictions)}")
        print(f"Total ground truth intervals: {len(self.ground_truth_intervals)}")
        print(f"Videos with predictions: {len(self.video_frames)}")
        print(f"Prediction file mappings: {len(self.pred_file_to_video)}")
        print(f"Activities in trained_classes: {len(all_activities)}")
        print(f"Activities in ground truth: {len(gt_activities)}")
        if extra_activities:
            print(f"Extra activities in GT (ignored): {extra_activities}")
            
        # Additional diagnostics
        self.print_detailed_diagnostics()
        
        return {
            "Precision per class (%)": {cls: f"{precision[cls]:.2f}%" for cls in precision},
            "Recall per class (%)": {cls: f"{recall[cls]:.2f}%" for cls in recall},
            "Overall Precision": f"{overall_precision:.2f}%",
            "Overall Recall": f"{overall_recall:.2f}%",
            "Mean IoU": f"{mean_iou*100:.2f}%"
        }
    
    def print_detailed_diagnostics(self):
        """Print detailed diagnostics to identify issues"""
        print(f"\n=== DETAILED DIAGNOSTICS ===")
        
        # Check frame coverage for each video  
        total_gt_frames = 0
        total_covered_frames = 0
        
        for video_path, frame_data in self.gt_frame_labels.items():
            gt_frames = set(frame_data.keys())
            covered_frames = set(frame for frame, data in frame_data.items() if data['pred_labels'])
            
            coverage = len(covered_frames) / len(gt_frames) if gt_frames else 0
            total_gt_frames += len(gt_frames)
            total_covered_frames += len(covered_frames)
            
            print(f"\nVideo: {Path(video_path).name}")
            print(f"  GT frames: {len(gt_frames)}")
            print(f"  Frames with predictions: {len(covered_frames)} (coverage: {coverage:.2%})")
        
        overall_coverage = total_covered_frames / total_gt_frames if total_gt_frames > 0 else 0
        print(f"\nOverall frame coverage: {overall_coverage:.2%}")
        
        # Check class distribution
        print(f"\n=== CLASS DISTRIBUTION (with interpolation) ===")
        
        pred_class_counts = {}
        gt_class_counts = {}
        all_activities = set(load_trained_classes("./src/trained_classes.txt"))
        
        for video_path, frame_data in self.gt_frame_labels.items():
            for frame, data in frame_data.items():
                # Count predictions
                for label in data['pred_labels']:
                    pred_class_counts[label] = pred_class_counts.get(label, 0) + 1
                
                # Count ground truth
                for label in data['gt_labels']:
                    if label in all_activities:
                        gt_class_counts[label] = gt_class_counts.get(label, 0) + 1
        
        print("Class distribution (frame-level with interpolation):")
        for cls in sorted(all_activities):
            pred_count = pred_class_counts.get(cls, 0)
            gt_count = gt_class_counts.get(cls, 0)
            ratio = pred_count / gt_count if gt_count > 0 else float('inf') if pred_count > 0 else 0
            print(f"  {cls}: Pred={pred_count}, GT={gt_count}, Ratio={ratio:.2f}")
        
        # Check for zero-prediction classes
        zero_pred_classes = []
        for cls in all_activities:
            if gt_class_counts.get(cls, 0) > 0 and pred_class_counts.get(cls, 0) == 0:
                zero_pred_classes.append(cls)
        
        if zero_pred_classes:
            print(f"\nClasses with zero predictions (causing 0% recall): {zero_pred_classes}")
        
        # Check prediction file mapping accuracy
        print(f"\n=== PREDICTION FILE MAPPING ===")
        for pred_file, video_path in self.pred_file_to_video.items():
            print(f"  {pred_file} -> {Path(video_path).name}")
            
            video_gt_count = len(self.ground_truth_intervals[
                self.ground_truth_intervals['video_path'] == video_path
            ])
            pred_count = len(self.predictions[self.predictions['pred_file'] == pred_file])
            print(f"    GT intervals: {video_gt_count}, Predictions: {pred_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_log", default="./inference_logs", help="Path to the prediction log folder")
    parser.add_argument("--output_file", default="./metrics_body", help="Path to save the evaluation results")
    parser.add_argument("--root_dir", default="./dataset/dmd/gA/5", type=str, help="Path to dataset root")
    args = parser.parse_args()

    pairs = collect_video_annotation_pairs(args.root_dir)
    metrics = Metric(args.prediction_log, pairs)
    results = metrics.evaluate()

    with open(args.output_file, 'w') as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    print(f"Metrics evaluation saved to {args.output_file}")