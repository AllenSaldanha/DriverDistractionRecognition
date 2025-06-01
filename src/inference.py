import cv2, os
cap = cv2.VideoCapture('./dataset/dmd/ga/1/s1/gA_1_s1_2019-03-08T09;31;15+01;00_ir_body.mp4')
os.makedirs('frames', exist_ok=True)
fps = cap.get(cv2.CAP_PROP_FPS)
idx = 0
while True:
    ret, frame = cap.read()
    if not ret: break
    # cv2.imwrite(f'frames/frame_{idx:05d}.jpg', frame)
    print(frame)
    idx += 1
cap.release()
