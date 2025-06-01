from inference import FrameLoader
from pose_detector import YOLOv5Detector
import cv2

def main():
    video_path = "./dataset/dmd/gA/1/s1/gA_1_s1_2019-03-08T09;31;15+01;00_rgb_body.mp4"
    loader = FrameLoader(video_path)
    detector = YOLOv5Detector()
    
    for frame in loader:
        results = detector.predict(frame)
        keypoints = detector.get_keypoints(results)
        
        if keypoints is not None:
            for person_kpts in keypoints.xy:
                print("Keypoints:", person_kpts.cpu().numpy().tolist())  # Convert to list for printing
        frame = detector.draw_results(frame, results)
        cv2.imshow("YOLOv5 Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    loader.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()