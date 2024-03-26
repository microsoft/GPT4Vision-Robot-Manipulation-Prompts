import vlm.high_level_planning.task_planner as task_planner
import vlm.video_analyzer as video_analyzer
import vlm.scene_analyzer as scene_analyzer
import argparse
import json
import os
import cv2
import base64
import numpy as np

out_dir = './output'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

def image_resize(frame):
    height, width = frame.shape[:2]
    aspect_ratio = width / height
    max_short_side = 768
    max_long_side = 2000
    if aspect_ratio > 1:  # Landscape orientation
        new_width = min(width, max_long_side)
        new_height = int(new_width / aspect_ratio)
        if new_height > max_short_side:
            new_height = max_short_side
            new_width = int(new_height * aspect_ratio)
    else:  # Portrait orientation
        new_height = min(height, max_long_side)
        new_width = int(new_height * aspect_ratio)
        if new_width > max_short_side:
            new_width = max_short_side
            new_height = int(new_width / aspect_ratio)
    resized_frame = cv2.resize(frame, (new_width, new_height))
    return resized_frame


def read_frames(video_path, frame_num=10):
    if not os.path.exists(video_path):
        print("Error: Video file not found.")
        return

    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_num > 1:
        skip = max(1, (total_frames - 1) // (frame_num - 1))
    else:
        skip = total_frames
    base64Frames = []
    for f in range(frame_num - 1):
        frame_id = f * skip
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        success, frame = video.read()
        if not success:
            break
        frame = image_resize(frame)
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
    video.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
    success, frame = video.read()
    if success:
        frame = image_resize(frame)
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
    video.release()
    print(len(base64Frames), "frames read.")
    return base64Frames


def concat_and_save_images(frames, file_name):
    images = [base64.b64decode(frame) for frame in frames]
    images = [
        cv2.imdecode(
            np.frombuffer(
                image,
                np.uint8),
            cv2.IMREAD_COLOR) for image in images]
    concat_image = np.concatenate(images, axis=1)
    cv2.imwrite(os.path.join(out_dir, file_name), concat_image)


def main(video_path, use_azure, frame_num=10, file_name='test'):
    selected_frames = read_frames(video_path, frame_num=frame_num)
    concat_and_save_images(
        selected_frames,
        file_name +
        '_recognized_frames.jpg')
    if use_azure:
        print("Using Azure OpenAI API.")
    print("Understanding video content...")
    textual_instruction = video_analyzer.video_understanding(
        selected_frames, use_azure=use_azure)
    print("Textual instruction:", textual_instruction)
    print("Understanding scene...")
    environmental_description = scene_analyzer.scene_understanding(
        selected_frames[0], textual_instruction, use_azure=use_azure)
    print(
        "Environmental description:",
        json.dumps(
            environmental_description,
            indent=4))
    print("Generating task plan...")
    aimodel = task_planner.planner(use_azure=use_azure)
    task_plan = aimodel.generate(
        textual_instruction,
        environmental_description)
    print("Task plan:", task_plan['task_cohesion']['task_sequence'])
    fp = os.path.join(out_dir, file_name + '_high-level_tasks.json')
    with open(fp, 'w') as f:
        json.dump(task_plan, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Understand the video content and generate a textual instruction.")
    parser.add_argument("video_path", type=str, help="Path to the video file.")
    parser.add_argument(
        "--use_azure",
        action="store_true",
        help="Use Azure OpenAI API to generate textual instruction.")
    args = parser.parse_args()
    frame_num = 5  # Number of frames to be analyzed.
    file_name = os.path.splitext(os.path.basename(args.video_path))[0] # Prefix for the output files.
    main(args.video_path, args.use_azure, frame_num=frame_num, file_name=file_name)
