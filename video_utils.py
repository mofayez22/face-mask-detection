import cv2
import os
import tempfile
from inference import run_inference_on_frame
import platform
from analytics_utils import analyze_detections

def init_video_stats():
    return {
        "total_frames": 0,
        "frames_with_detections": 0,
        "total_detections": 0,
        "with_mask": 0,
        "without_mask": 0,      
    }

def process_video(
    model,
    input_video_path,
    conf,
    fourcc,
    suffix,
    progress_bar=None,
    status_text=None
):
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, f"output{suffix}")

    cap = cv2.VideoCapture(input_video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = None

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("❌ Failed to read input video")

    height, width = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    video_stats = init_video_stats()
    frame_idx = 0

    while True:
        if frame_idx == 0:
            current_frame = frame
        else:
            ret, current_frame = cap.read()
            if not ret:
                break

        annotated, results = run_inference_on_frame(model, current_frame, conf)

        frame_stats = analyze_detections(results, conf)

        video_stats["total_frames"] += 1
        video_stats["total_detections"] += frame_stats["total_detections"]
        video_stats["with_mask"] += frame_stats["with_mask"]
        video_stats["without_mask"] += frame_stats["without_mask"]

        if frame_stats["total_detections"] > 0:
            video_stats["frames_with_detections"] += 1

        out.write(annotated.astype("uint8"))
        frame_idx += 1

        if progress_bar and total_frames:
            progress_bar.progress(min(frame_idx / total_frames, 1.0))

        if status_text:
            status_text.text(
                f"Processing frame {frame_idx}" +
                (f" / {total_frames}" if total_frames else "")
            )
    compliance = (video_stats["with_mask"] / video_stats["total_detections"]) * 100

    cap.release()
    out.release()

    return output_path, video_stats, compliance



"""import cv2
import os
from inference import run_inference_on_frame


def process_video_locally(model, input_video_path, conf):
    os.makedirs("outputs", exist_ok=True)

    output_path = os.path.abspath("outputs/output.avi")
    print("Saving output to:", output_path)

    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        raise RuntimeError("❌ Cannot open input video")

    # ---- Read first frame ----
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("❌ Failed to read first frame")

    height, width = frame.shape[:2]
    writer_size = (width, height)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1:
        fps = 25

    print("Video properties:")
    print(" - Size:", writer_size)
    print(" - FPS:", fps)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, fps, writer_size)

    if not out.isOpened():
        raise RuntimeError("❌ VideoWriter failed to open")

    frame_count = 0

    # ---- Process frames ----
    while True:
        if frame_count == 0:
            current_frame = frame
        else:
            ret, current_frame = cap.read()
            if not ret:
                break

        annotated = run_inference_on_frame(model, current_frame, conf)

        # FORCE size
        if annotated.shape[:2] != (height, width):
            annotated = cv2.resize(annotated, writer_size)

        annotated = annotated.astype("uint8")

        out.write(annotated)
        frame_count += 1

        if frame_count % 10 == 0:
            print(f"Written {frame_count} frames")

    cap.release()
    out.release()

    print("Total frames written:", frame_count)

    if frame_count == 0:
        raise RuntimeError("❌ Zero frames written")

    return output_path
"""