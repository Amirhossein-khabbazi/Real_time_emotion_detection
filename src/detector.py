import cv2
import os
import time
import threading
import logging
import numpy as np
from imutils.video import VideoStream, FPS
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.vs = None
        self.detecting = False
        self.photo_count = 0
        self.thread = None
        self.new_frame = None
        self.save_dir = r"./../photos"
        os.makedirs(self.save_dir, exist_ok=True)

        self.class_names = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprised']
        self.class_colors = {
            'Anger': (0, 0, 255), 'Contempt': (255, 0, 0), 'Disgust': (0, 255, 0),
            'Fear': (0, 255, 255), 'Happy': (255, 165, 0), 'Neutral': (255, 255, 255),
            'Sad': (255, 0, 255), 'Surprised': (0, 0, 0)
        }

    def start_detection(self):
        if not self.detecting:
            self.vs = VideoStream(src=0 + cv2.CAP_DSHOW).start()
            self.vs.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.vs.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.detecting = True
            self.thread = threading.Thread(target=self.detect_objects)
            self.thread.start()
            logging.info("Started detection")

    def stop_detection(self):
        if self.detecting:
            self.detecting = False
            self.thread.join()
            self.vs.stop()
            cv2.destroyAllWindows()
            logging.info("Stopped detection")

    def capture_photo(self):
        if self.detecting and self.new_frame is not None:
            photo_path = os.path.join(self.save_dir, f'captured_photo_{self.photo_count}.jpg')
            cv2.imwrite(photo_path, self.new_frame)
            logging.info(f'Photo captured and saved as {photo_path}')
            self.photo_count += 1
            return photo_path
        return None

    def non_max_suppression(self, boxes, overlap_thresh):
        if len(boxes) == 0:
            return []

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        pick = []
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))

        return boxes[pick].astype("int")

    def detect_objects(self):
        fps = None
        max_fps = 60
        frame_time = 1.0 / max_fps

        while self.detecting:
            start_time = time.time()
            frame = self.vs.read()
            if frame is None:
                logging.error('Failed to read frame from camera.')
                break

            results = self.model(frame, conf=0.50)  # Perform inference
            detected_boxes = []
            self.new_frame = frame.copy()

            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    detected_class = result.names[class_id]
                    color = self.class_colors.get(detected_class, (255, 0, 0))
                    detected_boxes.append([
                        int(box.xyxy[0][0]), int(box.xyxy[0][1]), 
                        int(box.xyxy[0][2]), int(box.xyxy[0][3]), 
                        detected_class, color, box.conf[0]
                    ])

            # Apply non-max suppression
            if detected_boxes:
                boxes = np.array([box[:4] for box in detected_boxes])
                nms_boxes = self.non_max_suppression(boxes, overlap_thresh=0.3)
                for i, (x1, y1, x2, y2) in enumerate(nms_boxes):
                    detected_class = detected_boxes[i][4]
                    color = detected_boxes[i][5]
                    confidence = detected_boxes[i][6]
                    cv2.rectangle(self.new_frame, (x1, y1), (x2, y2), color, 1)
                    cv2.putText(self.new_frame, f"{detected_class}: {confidence:.2f}",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)

            if fps is None:
                fps = FPS().start()
            else:
                fps.update()
                fps.stop()
                text = f"FPS: {fps.fps():.2f}"
                cv2.putText(self.new_frame, text, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow("Cam", self.new_frame)
            key = cv2.waitKey(1) & 0xFF

            elapsed_time = time.time() - start_time
            if elapsed_time < frame_time:
                time.sleep(frame_time - elapsed_time)

        cv2.destroyAllWindows()
