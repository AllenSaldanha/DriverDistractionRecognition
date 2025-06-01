from inference import FrameLoader
import cv2

def main():
    video_path = "./dataset/dmd/gA/1/s1/gA_1_s1_2019-03-08T09;31;15+01;00_rgb_body.mp4"
    loader = FrameLoader(video_path)
    
    for frame in loader:
        cv2.imshow("Driver Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    loader.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
