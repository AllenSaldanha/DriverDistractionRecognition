import os
from glob import glob

def collect_video_annotation_pairs(root_dir):
    """
    Scans the given root directory recursively and returns a list of tuples:
    (rgb_face_video_path, rgb_ann_distraction_json_path)
    Only returns pairs where both files exist and match in prefix.
    """
    video_json_pairs = []

    # Recursively search for all *_rgb_face.mp4 files
    rgb_face_files = glob(os.path.join(root_dir, '**', '*_rgb_body.mp4'), recursive=True)

    for video_path in rgb_face_files:
        # Derive the expected annotation file by replacing suffix
        base_path = video_path.replace('_rgb_body.mp4', '')
        annotation_path = base_path + '_rgb_ann_distraction.json'

        if os.path.exists(annotation_path):
            video_json_pairs.append((video_path, annotation_path))
        else:
            print(f"[WARN] Annotation file not found for: {video_path}")

    return video_json_pairs
