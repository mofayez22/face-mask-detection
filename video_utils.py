import cv2
import os
import tempfile
from inference import run_inference_on_frame
import platform



def process_video(model, input_video_path, conf):
    
    if platform.system() == "Windows":
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        suffix = ".avi"
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        suffix = ".mp4"

    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, f"output{suffix}")

    cap = cv2.VideoCapture(input_video_path)

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("❌ Failed to read input video")

    height, width = frame.shape[:2]
    writer_size = (width, height)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 25

    
    

    out = cv2.VideoWriter(output_path, fourcc, fps, writer_size)

    if not out.isOpened():
        raise RuntimeError("❌ VideoWriter failed to open")

    frame_count = 0

    while True:
        if frame_count == 0:
            current_frame = frame
        else:
            ret, current_frame = cap.read()
            if not ret:
                break

        annotated = run_inference_on_frame(model, current_frame, conf)

        if annotated.shape[:2] != (height, width):
            annotated = cv2.resize(annotated, writer_size)

        annotated = annotated.astype("uint8")
        out.write(annotated)
        frame_count += 1

    cap.release()
    out.release()

    if frame_count == 0:
        raise RuntimeError("❌ No frames written")

    return output_path




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