import cv2
import numpy as np
from object_detection import ObjectDetection
import math
import torch

od = ObjectDetection()

cap = cv2.VideoCapture("football_clip.mp4")

# Initialize count
count = 0
previous_frame_points = []

tracking_objects = {}
track_id = 0

def calculate_distance(point1, point2):
    return math.hypot(point2[0] - point1[0], point2[1] - point1[1])

def assign_initlal_ids(previous_frame_points, current_frame_points, track_id):
    '''this function will take the previous frame and current frame to see if the vehicle is the same and assign a new id'''
    for current in current_frame_points:
            for previous in previous_frame_points:
                distance = math.hypot(previous[0] - current[0], previous[1] - current[1])

                if distance < 20:
                    tracking_objects[track_id] = current
                    track_id += 1

    return tracking_objects, track_id

def update_distance_and_delete(current_frame_points, current_frame_points_copy, tracking_objects_copy):
    for object_id, pt2 in tracking_objects_copy.items():
        object_exists = False
        for pt in current_frame_points_copy:
            distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

            if distance < 20:
                tracking_objects[object_id] = pt
                object_exists = True
                if pt in current_frame_points:
                    current_frame_points.remove(pt) #remove the points which you updated so the unremoved points can be assigned new ids
                continue

        if not object_exists:
            tracking_objects.pop(object_id)
    
    return tracking_objects

while True:
    ret, frame = cap.read()
    count += 1
    if not ret:
        break

    current_frame_points = []

    (class_ids, scores, boxes) = od.detect(frame)  
    for box in boxes:
        (x, y, w, h) = box
        cx = int((x + x + w) / 2)
        cy = int((y + y + h) / 2)
        current_frame_points.append((cx, cy))

        # cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Only at the beginning we compare previous and current frame to get some new ids for each object
    if count <= 2:
        tracking_objects, track_id = assign_initlal_ids(previous_frame_points, current_frame_points, track_id)
    else:
        tracking_objects_copy = tracking_objects.copy()
        current_frame_points_copy = current_frame_points.copy()

        tracking_objects = update_distance_and_delete(current_frame_points, current_frame_points_copy, tracking_objects_copy)
    
        # adding new ids for the unassigned objects
        for pt in current_frame_points:
            tracking_objects[track_id] = pt
            track_id += 1
    
    for object_id, pt in tracking_objects.items():
        cv2.circle(frame, pt, 5, (0, 0, 255), -1)
        cv2.putText(frame, str(object_id), (pt[0], pt[1] - 7), 0, 1, (0, 0, 255), 2)

    print("Tracking objects")
    print(tracking_objects)

    cv2.imshow("frame", frame)

    previous_frame_points = current_frame_points.copy()

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()










