from inference import FrameLoader
from pose_detector import YOLOv8Detector
from activity_classifier import DriverActivityClassifier
import cv2
import torch

def main():
    video_path = "./dataset/dmd/gA/1/s1/gA_1_s1_2019-03-08T09;31;15+01;00_rgb_face.mp4"
    loader = FrameLoader(video_path)
    detector = YOLOv8Detector()
    classifier = DriverActivityClassifier()
    
    for frame in loader:
        results = detector.predict(frame)
        keypoints_list = detector.get_keypoints(results)  # Returns list of (17, 2) keypoints

        annotated = detector.draw_results(frame, results)

        if keypoints_list is not None:
            for i, kpts in enumerate(keypoints_list):
                kpts_np = kpts.cpu().numpy() if torch.is_tensor(kpts) else kpts
                activity = classifier.classify(kpts_np)
                print(activity)

                # Draw activity label on the annotated frame
                cv2.putText(
                    annotated, 
                    f"Person {i+1}: {activity}", 
                    (30, 60 + i * 40), 
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    3
                )

        cv2.imshow("Driver Activity Recognition", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    loader.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()