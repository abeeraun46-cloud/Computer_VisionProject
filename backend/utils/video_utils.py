import cv2
from backend.utils.behavior import BehaviorAnalyzer

def process_video(model, input_path, output_path):
    cap = cv2.VideoCapture(input_path)

    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    analyzer = BehaviorAnalyzer()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, classes=[0], conf=0.4)

        detections = []
        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            detections.append((x1, y1, x2, y2))

        analyzed = analyzer.update(detections)

        for obj in analyzed:
            x1, y1, x2, y2 = obj["bbox"]
            pid = obj["id"]
            status = obj["status"]

            color = (0,255,0) if status == "Normal" else (0,0,255)

            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, f"ID {pid} - {status}",
                        (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2)

        out.write(frame)

    cap.release()
    out.release()