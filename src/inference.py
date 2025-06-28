import argparse
from utils.video_annotation_pairs import collect_video_annotation_pairs

def inference(pair):
    # This function should load the model and perform inference on the video frames
    print(pair)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Driver Activity Inference")
    parser.add_argument("--root_dir", default = "./dataset/dmd/gA/2", type=str, help="Path to dataset root")
    args = parser.parse_args()

    pairs = collect_video_annotation_pairs(args.root_dir)
    print(f"Found {len(pairs)} valid video-annotation pairs.")

    inference(pairs)