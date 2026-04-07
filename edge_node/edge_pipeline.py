import cv2
import json
import os
import math
import glob
from itertools import combinations
from ultralytics import YOLO

# Load YOLOv8 Pose Model
model = YOLO('yolov8n-pose.pt')

VIDEO_DIR = "../data/dataset"
TRACKER_CONFIG = "../configs/custom_botsort.yaml"

def get_posture(w, h):
    aspect_ratio = h / w
    if aspect_ratio < 1.4:
        return "crouching"
    return "standing"

def detect_suspicious_action(keypoints, box_height):
    if len(keypoints) < 13:
        return False
        
    l_wrist, r_wrist = keypoints[9], keypoints[10]
    l_hip, r_hip = keypoints[11], keypoints[12]
    
    def is_hand_near_pocket(wrist, hip):
        if wrist[0] == 0 or hip[0] == 0: 
            return False
        dist = math.hypot(wrist[0] - hip[0], wrist[1] - hip[1])
        return dist < (box_height * 0.15)

    return is_hand_near_pocket(l_wrist, l_hip) or is_hand_near_pocket(r_wrist, r_hip)

def get_proximity_alerts(boxes, track_ids):
    alerts = []
    for i, j in combinations(range(len(boxes)), 2):
        x1, y1, w1, h1 = boxes[i]
        x2, y2, w2, h2 = boxes[j]
        
        dist = math.hypot(x1 - x2, y1 - y2)
        avg_width = (w1 + w2) / 2
        
        if dist < (avg_width * 0.85): 
            alerts.append(f"ID {track_ids[i]} and ID {track_ids[j]} are in extreme close proximity (touching).")
            
    return alerts

def run_robust_batch_pipeline():
    os.makedirs("../data", exist_ok=True)
    video_files = glob.glob(os.path.join(VIDEO_DIR, "*.mp4"))
    
    if not video_files:
        print(f"Error: No .mp4 files found in {VIDEO_DIR}.")
        return

    print(f"Robust Batch Pipeline Active. Found {len(video_files)} videos to process.")
    
    master_metadata = {
        "dataset_directory": VIDEO_DIR,
        "total_videos_processed": len(video_files),
        "processed_videos": []
    }

    for video_path in video_files:
        filename = os.path.basename(video_path)
        print(f"\n--- Processing: {filename} ---")
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = 0
        CHUNK_SIZE = int(fps / 2) 
        
        # NEW: Track Lifespan Dictionary to prevent Ghost IDs
        id_lifespan = {}
        validated_unique_ids = set()
        
        video_data = {
            "filename": filename,
            "total_unique_people": 0,
            "active_ids": [],
            "chunks": []
        }

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # NEW: classes=[0] forces it to ONLY look for humans. 
            # NEW: conf=0.40 ignores low-confidence noise (windows/chairs).
            results = model.track(frame, persist=True, tracker=TRACKER_CONFIG, iou=0.45, conf=0.40, classes=[0], verbose=False)

            if frame_count % CHUNK_SIZE == 0:
                chunk_num = frame_count // CHUNK_SIZE
                current_active_ids = []
                frame_actions = []
                proximity_alerts = []

                if results[0].boxes is not None and results[0].boxes.id is not None and results[0].keypoints is not None:
                    boxes = results[0].boxes.xywh.cpu().numpy() 
                    track_ids = results[0].boxes.id.int().cpu().numpy()
                    keypoints_array = results[0].keypoints.xy.cpu().numpy()
                    
                    proximity_alerts = get_proximity_alerts(boxes, track_ids)

                    for box, track_id, kpts in zip(boxes, track_ids, keypoints_array):
                        t_id = int(track_id)
                        
                        # NEW: Increment the lifespan counter for this ID
                        id_lifespan[t_id] = id_lifespan.get(t_id, 0) + 1
                        
                        # NEW: Only count them as a "real" person if they survive 10 tracking cycles
                        if id_lifespan[t_id] > 10:
                            validated_unique_ids.add(t_id)
                            
                        current_active_ids.append(t_id)
                        
                        _, _, w, h = box
                        action = get_posture(w, h)
                        
                        if detect_suspicious_action(kpts, h):
                            action = "hiding items in clothing"
                        
                        frame_actions.append({
                            "id": t_id,
                            "status": "SUSPICIOUS" if action in ["crouching", "hiding items in clothing"] else "NORMAL",
                            "action": action
                        })
                
                chunk_data = {
                    "chunk_id": chunk_num,
                    "people_in_frame": len(current_active_ids),
                    "active_ids": current_active_ids,
                    "tracking_data": frame_actions,
                    "proximity_flags": proximity_alerts
                }
                video_data["chunks"].append(chunk_data)

            # UI Visualization
            annotated_frame = results[0].plot()
            cv2.putText(annotated_frame, f"Validated Unique People: {len(validated_unique_ids)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Robust Enterprise Pipeline", annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print(f"Skipping the rest of {filename}...")
                break

        # Finalize data using ONLY the validated IDs
        video_data["total_unique_people"] = len(validated_unique_ids)
        video_data["active_ids"] = list(validated_unique_ids)
        
        master_metadata["processed_videos"].append(video_data)
        cap.release()

    cv2.destroyAllWindows()
    log_path = "../data/master_metadata_log.json"
    with open(log_path, "w") as f:
        json.dump(master_metadata, f, indent=4)
            
    print(f"\n=====================================")
    print(f"BATCH COMPLETE! {len(video_files)} videos processed.")
    print(f"Master JSON saved to {log_path}")

if __name__ == "__main__":
    run_robust_batch_pipeline()