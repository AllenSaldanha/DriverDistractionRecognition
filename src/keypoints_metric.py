import argparse
import json
import pandas as pd
import torch
import numpy as np
from pathlib import Path
from keypoint_dataset import DriverActivityKeypointDataset, load_trained_classes
from models.LSTM import (KeypointLSTM, KeypointGRU, KeypointTransformer, 
                         KeypointCNN1D, KeypointAttentionLSTM)
from utils.video_annotation_pairs import collect_video_annotation_pairs
from torch.utils.data import DataLoader

class KeypointMetrics:
    def __init__(self, model_path, keypoints_folder, pairs, device='cuda'):
        self.pairs = pairs
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.keypoints_folder = keypoints_folder
        
        # Load model and get predictions
        self.model, self.dataset = self.load_model_and_dataset(model_path)
        self.predictions = self.generate_predictions()
        
        # Load ground truth
        self.action_classes = load_trained_classes("./src/trained_classes.txt")
        self.ground_truth_intervals = self.load_ground_truth(pairs)
        
        # Convert ground truth intervals to frame labels
        self.gt_frame_labels = self.gt_intervals_to_frame_labels()

    def load_model_and_dataset(self, model_path):
        """Load trained model and create dataset"""
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create dataset
        dataset = DriverActivityKeypointDataset(
            keypoints_folder=self.keypoints_folder,
            video_annotation_pairs=self.pairs,
            num_frames=16
        )
        
        # Initialize model based on saved configuration
        model_type = checkpoint['model_type']
        num_classes = checkpoint['num_classes']
        args = checkpoint.get('args')
        
        model_dict = {
            'lstm': KeypointLSTM,
            'gru': KeypointGRU,
            'transformer': KeypointTransformer,
            'cnn1d': KeypointCNN1D,
            'attention_lstm': KeypointAttentionLSTM
        }
        
        model_class = model_dict[model_type]
        
        if model_type == 'transformer':
            model = model_class(
                num_classes=num_classes,
                max_persons=1,
                d_model=args.hidden_size if args else 128,
                nhead=8,
                num_layers=args.num_layers if args else 2,
                dropout=args.dropout if args else 0.3
            )
        else:
            model = model_class(
                num_classes=num_classes,
                max_persons=1,
                hidden_size=args.hidden_size if args else 128,
                num_layers=args.num_layers if args else 2,
                dropout=args.dropout if args else 0.3
            )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        print(f"Loaded {model_type} model with {num_classes} classes")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return model, dataset

    def generate_predictions(self):
        """Generate predictions for all sequences in the dataset"""
        print("Generating predictions...")
        
        batch_size = 8
        dataloader = DataLoader(
            self.dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True
        )
        
        all_predictions = []
        sample_idx = 0
        
        with torch.no_grad():
            for _, (keypoints, _) in enumerate(dataloader):
                keypoints = keypoints.to(self.device, non_blocking=True)
                
                # Get model predictions
                outputs = self.model(keypoints)
                probabilities = torch.sigmoid(outputs)
                
                # Apply threshold to get binary predictions
                threshold = 0.5
                predictions = (probabilities > threshold).cpu().numpy()
                
                # Store predictions with metadata
                current_batch_size = keypoints.shape[0]
                for i in range(current_batch_size):
                    if sample_idx < len(self.dataset.samples):
                        keypoints_dir, ann_path, start_frame, _ = self.dataset.samples[sample_idx]
                        video_name = keypoints_dir.name
                        
                        # Get predicted class names
                        pred_indices = np.where(predictions[i])[0]
                        pred_classes = [self.dataset.action_classes[idx] for idx in pred_indices]
                        
                        all_predictions.append({
                            'video_name': video_name,
                            'start_frame': start_frame,
                            'end_frame': start_frame + 15,  # 16 frames sequence
                            'predicted_classes': pred_classes,
                            'probabilities': probabilities[i].cpu().numpy()
                        })
                    
                    sample_idx += 1
        
        print(f"Generated {len(all_predictions)} predictions")
        return all_predictions

    def load_ground_truth(self, pairs):
        """Load ground truth intervals from annotation files"""
        gt_records = []
        for video_path, ann_path in pairs:
            video_name = Path(video_path).stem
            
            with open(ann_path, 'r') as f:
                try:
                    actions = json.load(f)["openlabel"].get("actions", {})
                except Exception as e:
                    print(f"Failed to load {ann_path}: {e}")
                    continue
                    
            for _, info in actions.items():
                activity = info.get("type")
                if activity in self.action_classes:  # Only include known activities
                    for interval in info.get("frame_intervals", []):
                        gt_records.append({
                            "frame_start": interval["frame_start"],
                            "frame_end": interval["frame_end"],
                            "activity": activity,
                            "video_name": video_name,
                            "video_path": video_path
                        })
        
        return pd.DataFrame(gt_records)

    def gt_intervals_to_frame_labels(self):
        """Convert ground truth intervals to frame-level labels for evaluation"""
        gt_frame_labels = {}
        
        # Group predictions by video
        video_predictions = {}
        for pred in self.predictions:
            video_name = pred['video_name']
            if video_name not in video_predictions:
                video_predictions[video_name] = []
            video_predictions[video_name].append(pred)
        
        # Process each video
        for video_name in video_predictions.keys():
            # Get ground truth for this video
            video_gt = self.ground_truth_intervals[
                self.ground_truth_intervals['video_name'] == video_name
            ]
            
            if len(video_gt) == 0:
                continue
                
            # Get all frames covered by ground truth
            gt_frames = set()
            for _, row in video_gt.iterrows():
                gt_frames.update(range(row['frame_start'], row['frame_end'] + 1))
            
            # Get predictions for this video
            video_preds = video_predictions[video_name]
            
            # Create frame-level labels
            gt_frame_labels[video_name] = {}
            
            for frame in gt_frames:
                # Ground truth labels for this frame
                gt_labels = set()
                for _, row in video_gt.iterrows():
                    if row['frame_start'] <= frame <= row['frame_end']:
                        gt_labels.add(row['activity'])
                
                # Find overlapping predictions
                pred_labels = set()
                for pred in video_preds:
                    if pred['start_frame'] <= frame <= pred['end_frame']:
                        pred_labels.update(pred['predicted_classes'])
                
                gt_frame_labels[video_name][frame] = {
                    'gt_labels': gt_labels,
                    'pred_labels': pred_labels
                }
        
        return gt_frame_labels

    def evaluate_multiclass(self):
        """Evaluate multi-class classification metrics"""
        metrics = {cls: {'tp': 0, 'fp': 0, 'fn': 0} for cls in self.action_classes}
        
        total_frames = 0
        for video_name, frame_data in self.gt_frame_labels.items():
            for frame, data in frame_data.items():
                total_frames += 1
                pred_labels = data['pred_labels']
                gt_labels = data['gt_labels']
                
                for cls in self.action_classes:
                    pred_has = cls in pred_labels
                    gt_has = cls in gt_labels
                    
                    if pred_has and gt_has:
                        metrics[cls]['tp'] += 1
                    elif pred_has and not gt_has:
                        metrics[cls]['fp'] += 1
                    elif not pred_has and gt_has:
                        metrics[cls]['fn'] += 1
        
        # Calculate precision and recall per class
        precision, recall, f1_score = {}, {}, {}
        for cls in self.action_classes:
            tp, fp, fn = metrics[cls]['tp'], metrics[cls]['fp'], metrics[cls]['fn']
            
            precision[cls] = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
            recall[cls] = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
            
            if precision[cls] + recall[cls] > 0:
                f1_score[cls] = 2 * (precision[cls] * recall[cls]) / (precision[cls] + recall[cls])
            else:
                f1_score[cls] = 0
        
        # Calculate overall metrics
        overall_tp = sum(metrics[cls]['tp'] for cls in self.action_classes)
        overall_fp = sum(metrics[cls]['fp'] for cls in self.action_classes)
        overall_fn = sum(metrics[cls]['fn'] for cls in self.action_classes)
        
        overall_precision = overall_tp / (overall_tp + overall_fp) * 100 if (overall_tp + overall_fp) > 0 else 0
        overall_recall = overall_tp / (overall_tp + overall_fn) * 100 if (overall_tp + overall_fn) > 0 else 0
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
        
        return precision, recall, f1_score, overall_precision, overall_recall, overall_f1

    def calculate_iou(self):
        """Calculate Intersection over Union (IoU) for multi-label predictions"""
        iou_scores = []
        
        for video_name, frame_data in self.gt_frame_labels.items():
            for frame, data in frame_data.items():
                pred_labels = data['pred_labels']
                gt_labels = data['gt_labels']
                
                intersection = len(pred_labels.intersection(gt_labels))
                union = len(pred_labels.union(gt_labels))
                
                if union > 0:
                    iou = intersection / union
                else:
                    iou = 1.0 if len(pred_labels) == 0 and len(gt_labels) == 0 else 0.0
                
                iou_scores.append(iou)
        
        return sum(iou_scores) / len(iou_scores) if iou_scores else 0.0

    def evaluate(self):
        """Main evaluation function"""
        print("Evaluating keypoint model performance...")
        
        precision, recall, f1_score, overall_precision, overall_recall, overall_f1 = self.evaluate_multiclass()
        mean_iou = self.calculate_iou()
        
        # Print results
        print("\n" + "="*50)
        print("KEYPOINT MODEL EVALUATION RESULTS")
        print("="*50)
        
        print(f"\nOverall Metrics:")
        print(f"  Precision: {overall_precision:.2f}%")
        print(f"  Recall: {overall_recall:.2f}%")
        print(f"  F1-Score: {overall_f1:.2f}%")
        print(f"  Mean IoU: {mean_iou*100:.2f}%")
        
        print(f"\nPer-Class Metrics:")
        print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-" * 60)
        for cls in sorted(self.action_classes):
            print(f"{cls:<20} {precision[cls]:<12.2f} {recall[cls]:<12.2f} {f1_score[cls]:<12.2f}")
        
        # Print diagnostic information
        self.print_diagnostics()
        
        return {
            "Overall Precision": f"{overall_precision:.2f}%",
            "Overall Recall": f"{overall_recall:.2f}%",
            "Overall F1-Score": f"{overall_f1:.2f}%",
            "Mean IoU": f"{mean_iou*100:.2f}%",
            "Per-Class Precision": {cls: f"{precision[cls]:.2f}%" for cls in self.action_classes},
            "Per-Class Recall": {cls: f"{recall[cls]:.2f}%" for cls in self.action_classes},
            "Per-Class F1-Score": {cls: f"{f1_score[cls]:.2f}%" for cls in self.action_classes}
        }

    def print_diagnostics(self):
        """Print detailed diagnostic information"""
        print(f"\n" + "="*50)
        print("DIAGNOSTIC INFORMATION")
        print("="*50)
        
        # Dataset statistics
        print(f"Total predictions generated: {len(self.predictions)}")
        print(f"Total ground truth intervals: {len(self.ground_truth_intervals)}")
        print(f"Videos processed: {len(self.gt_frame_labels)}")
        
        # Frame coverage statistics
        total_gt_frames = sum(len(frame_data) for frame_data in self.gt_frame_labels.values())
        total_pred_frames = 0
        covered_frames = 0
        
        for video_name, frame_data in self.gt_frame_labels.items():
            for frame, data in frame_data.items():
                total_pred_frames += 1
                if data['pred_labels']:
                    covered_frames += 1
        
        coverage = covered_frames / total_gt_frames if total_gt_frames > 0 else 0
        print(f"Frame coverage: {coverage:.2%} ({covered_frames}/{total_gt_frames})")
        
        # Class distribution
        print(f"\nClass Distribution:")
        pred_class_counts = {}
        gt_class_counts = {}
        
        for video_name, frame_data in self.gt_frame_labels.items():
            for frame, data in frame_data.items():
                for label in data['pred_labels']:
                    pred_class_counts[label] = pred_class_counts.get(label, 0) + 1
                for label in data['gt_labels']:
                    gt_class_counts[label] = gt_class_counts.get(label, 0) + 1
        
        print(f"{'Class':<20} {'GT Count':<12} {'Pred Count':<12} {'Ratio':<12}")
        print("-" * 60)
        for cls in sorted(self.action_classes):
            gt_count = gt_class_counts.get(cls, 0)
            pred_count = pred_class_counts.get(cls, 0)
            ratio = pred_count / gt_count if gt_count > 0 else float('inf') if pred_count > 0 else 0
            print(f"{cls:<20} {gt_count:<12} {pred_count:<12} {ratio:<12.2f}")
        
        # Video-wise statistics
        print(f"\nVideo-wise Coverage:")
        print(f"{'Video':<30} {'GT Frames':<12} {'Coverage':<12}")
        print("-" * 60)
        for video_name, frame_data in self.gt_frame_labels.items():
            gt_frames = len(frame_data)
            video_covered = sum(1 for data in frame_data.values() if data['pred_labels'])
            video_coverage = video_covered / gt_frames if gt_frames > 0 else 0
            print(f"{video_name[:28]:<30} {gt_frames:<12} {video_coverage:<12.2%}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Keypoint Model Performance")
    parser.add_argument("--model_path", default="./best_keypoint_body_lstm_model.pth", help="Path to trained model checkpoint")
    parser.add_argument("--keypoints_folder", default="./keypoints/metrics", help="Path to keypoints folder")
    parser.add_argument("--root_dir", default="./dataset/dmd/gA/5", help="Path to dataset root")
    parser.add_argument("--output_file", default="./keypoint_body_metrics_results_two.json", help="Output file for results")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device to use for evaluation")
    
    args = parser.parse_args()
    
    # Collect video-annotation pairs
    pairs = collect_video_annotation_pairs(args.root_dir)
    print(f"Found {len(pairs)} video-annotation pairs")
    
    if len(pairs) == 0:
        print("No valid video-annotation pairs found!")
        return
    
    # Initialize metrics evaluator
    evaluator = KeypointMetrics(
        model_path=args.model_path,
        keypoints_folder=args.keypoints_folder,
        pairs=pairs,
        device=args.device
    )
    
    # Run evaluation
    results = evaluator.evaluate()
    
    # Save results
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()