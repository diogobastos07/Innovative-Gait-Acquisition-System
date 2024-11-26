import os
from ultralytics import YOLO
import cv2
from collections import defaultdict
import numpy as np
from utils import load_config, get_color_for_id, get_msg_mgr
from classes import SQAM
import sys


# Function to annotate frames with bounding boxes, IDs, and tracking history
def draw_in_frame(frame, track_history, boxes, track_ids, confidences):
    for box, track_id, confidence in zip(boxes, track_ids, confidences):
        x, y, w, h = box
        track = track_history[track_id] # Retrieve or initialize track history for the ID
        track.append((float(x), float(y))) # Add the current position to the track history
        if len(track) > 30: # Limit track history to the last 30 points
            track.pop(0)

        # Convert bounding box coordinates to integers and draw rectangle on the frame
        x, y, w, h = map(int, (x, y, w, h))
        color = get_color_for_id(track_id) # Get consistent color for the track ID
        cv2.rectangle(frame, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), color, 2)
        
        # Add a label with ID and confidence
        label = f"ID: {track_id}; Conf: {confidence:.2f}"  
        cv2.putText(frame, label, (x - w // 2, y - h // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Draw tracking lines for the last points in the history
        if len(track) > 1:
            points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=color, thickness=2)
    return frame



if __name__ == '__main__':
    # Load configuration file and prepare output directory
    cfg_path = os.path.abspath('acquisition_system/configs/system.yaml')
    output_path = "outputs/acquisition_system/"
    cfg = load_config(cfg_path)

    # Extract configuration sections
    geral_cfg = cfg['geral_cfg']
    track_cfg = cfg['track_cfg']
    sqam_cfg = cfg['sqam_cfg']

    # Initialize logging system
    msg_mgr = get_msg_mgr()
    msg_mgr.init_logger(output_path, geral_cfg['log_to_file'])

    # Log configuration details
    msg_mgr.log_info(f'Config file loaded from {cfg_path}')
    msg_mgr.log_info(geral_cfg)
    msg_mgr.log_info(track_cfg)
    msg_mgr.log_info(sqam_cfg)

    # Load YOLO model and video input
    model = YOLO(geral_cfg['model_path'])
    cap = cv2.VideoCapture(geral_cfg['input_video_path'])

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    msg_mgr.log_info(f"Video Properties --> \'fps\': {fps}, \'width\': {width}, \'height\': {height}")

    # Setup video saving and/or frame display
    save_video = geral_cfg['save_annotated_video']
    show_frames = geral_cfg['show_annotated_frames']
    if save_video:
        os.makedirs(os.path.join(output_path, 'annotated_video'), exist_ok=True)
        output_path_video = os.path.join(output_path, 'annotated_video', geral_cfg['name'] + '.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path_video, fourcc, fps, (width, height))
    if save_video or show_frames: 
        track_history = defaultdict(lambda: []) # Initialize track histories for visualizations

    # Initialize Sequence Quality Analysis Module (SQAM) 
    try:
        sqam = SQAM(height, width, **sqam_cfg)
    except ValueError as e:
        msg_mgr.log_warning(f"Error while creating the SQAM: {e}")
        sys.exit()
    msg_mgr.log_info('Start Tracking!')
    msg_mgr.reset_time()
    
    # Video processing loop
    while cap.isOpened():
        success, frame = cap.read() # Read next video frame
        if success:
            # Perform object tracking with YOLO
            results = model.track(frame, persist = True, **track_cfg)
            boxes = results[0].boxes.xywh.cpu().numpy() # Get bounding boxes
            track_ids = results[0].boxes.id.int().cpu().tolist() # Get object IDs

            # Process current frame data in SQAM
            sqam.process_new_frame(frame, boxes, track_ids.copy())
            msg_mgr.log_system_info(sqam.tracking_dict, sqam.exclusion_dict, sqam.complete_sequence_dict)

            # Draw annotations if enabled
            if save_video or show_frames: 
                confidences=results[0].boxes.conf.cpu() # Get confidence scores
                frame = draw_in_frame(frame, track_history, boxes, track_ids, confidences)

            if save_video:
                out.write(frame) # Save the annotated frame to output video
            if show_frames:
                cv2.imshow("Tracking", frame) # Display the frame

            # Exit loop on 'q' key press
            if (cv2.waitKey(1) & 0xFF == ord("q")):
                break
        else:
            break # Stop loop if no more frames

    # Cleanup resources
    cap.release()
    if save_video:
        out.release()
        msg_mgr.log_info(f"Annotated video saved in {output_path_video}")
    sqam.end(output_path)
    cv2.destroyAllWindows()


