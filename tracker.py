import cv2
from ultralytics import YOLO
import time
   

def process_video(
    input_path,
    output_path="outputs/processed.mp4",
    model_path="models/yolov11s.pt",
    mode="zone",
    zone=None,
    line=None,
):
    t_start     = time.time()
    

    model  = YOLO(model_path)
    cap    = cv2.VideoCapture(input_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    cap.release()  # ← release after all cap.get() calls

    # defaults
    if mode == "zone" and zone is None:
        zone = (int(width*0.25), int(height*0.25), int(width*0.75), int(height*0.75))
    if mode == "line" and line is None:
        line = ((int(width*0.5), 0), (int(width*0.5), height))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    counted_ids = set()
    count       = 0
    previous_points = {}
    class_counts = {}  # ← per class counts
    

    results = model.track(
        source=input_path,
        tracker="bytetrack.yaml",
        stream=True,
        persist=True,
        conf=0.4,
        imgsz=640
    )

    for r in results:
        frame = r.orig_img.copy()

        if r.boxes.id is not None:
            boxes   = r.boxes.xyxy.cpu().numpy()
            ids     = r.boxes.id.cpu().numpy().astype(int)
            classes = r.boxes.cls.cpu().numpy().astype(int)

            for box, track_id, cls in zip(boxes, ids, classes):
                x1, y1, x2, y2 = map(int, box)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                cls_name = model.names[cls]
                label    = f"{cls_name} ID:{track_id}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

                if mode == "zone":
                    zone_x1, zone_y1, zone_x2, zone_y2 = zone
                    if (zone_x1 < cx < zone_x2 and
                        zone_y1 < cy < zone_y2 and
                        track_id not in counted_ids):
                        count += 1
                        counted_ids.add(track_id)
                        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

                elif mode == "line":
                    (lx1, ly1), (lx2, ly2) = line

                    prev_point = previous_points.get(track_id)

                    if prev_point is not None and track_id not in counted_ids:
                        prev_x, prev_y = prev_point

                        # signed distance from point to line
                        prev_side = (lx2 - lx1) * (prev_y - ly1) - (ly2 - ly1) * (prev_x - lx1)
                        curr_side = (lx2 - lx1) * (cy - ly1) - (ly2 - ly1) * (cx - lx1)

                        if prev_side * curr_side < 0:
                            count += 1
                            counted_ids.add(track_id)
                            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

                    previous_points[track_id] = (cx, cy)

        if mode == "zone":
            zone_x1, zone_y1, zone_x2, zone_y2 = zone
            cv2.rectangle(frame, (zone_x1, zone_y1), (zone_x2, zone_y2), (255, 0, 0), 2)
        elif mode == "line":
            (lx1, ly1), (lx2, ly2) = line
            cv2.line(frame, (lx1, ly1), (lx2, ly2), (255, 0, 0), 2)

        cv2.putText(frame, f"Count: {count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        out.write(frame)

    out.release()

    t_end          = time.time()
    inference_time = round(t_end - t_start, 2)
    return {"count": count, "output_path": output_path, "class_counts": class_counts, "Time taken":inference_time}