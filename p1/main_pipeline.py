import cv2
import numpy as np
import math
from ultralytics import YOLO


print("Loading YOLOv8s AI Model...")
model = YOLO('yolov8s.pt') 

video_path = r"C:\Users\anshu\OneDrive\Desktop\TLA\highway_test.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0



lane1_poly = np.array([[579, 346], [561, 304], [628, 297], [663, 336]], np.int32)
lane2_poly = np.array([[663, 335], [628, 297], [691, 293], [740, 328]], np.int32)
lane3_poly = np.array([[740, 329], [691, 294], [751, 287], [810, 318]], np.int32)
lane4_poly = np.array([[344, 343], [376, 304], [460, 308], [446, 350]], np.int32) 
lane5_poly = np.array([[264, 336], [309, 301], [376, 304], [344, 343]], np.int32) 
lane6_poly = np.array([[183, 322], [243, 294], [309, 301], [264, 336]], np.int32) 

polygons = [lane1_poly, lane2_poly, lane3_poly, lane4_poly, lane5_poly, lane6_poly]
colors = [(255, 0, 255), (255, 255, 0), (0, 165, 255), (0, 255, 0), (0, 255, 255), (147, 20, 255)] 

lane_counts = {i: {'in': 0, 'out': 0} for i in range(6)}
class_counts = {'Car': 0, 'Truck/Bus': 0, 'Moto': 0}
vehicle_colors = {}
vehicle_speeds = {}
counted_ids = set()

vehicle_history = {}        
vehicle_warped_history = {} 

class_map = {1: 'Moto', 2: 'Car', 3: 'Moto', 5: 'Truck/Bus', 7: 'Truck/Bus'}


src_pts = np.float32([
    [450, 200],   
    [650, 200],   
    [1020, 500],  
    [0, 500]      
])

flat_width, flat_height = 1000, 1000
dst_pts = np.float32([[0, 0], [flat_width, 0], [flat_width, flat_height], [0, flat_height]])


warp_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)


REAL_WORLD_METERS = 90.0
METER_PER_PIXEL = REAL_WORLD_METERS / flat_height


while True:
    success, frame = cap.read()
    if not success: break
    frame = cv2.resize(frame, (1020, 500))


    overlay = frame.copy()
    for i, poly in enumerate(polygons):
        cv2.fillPoly(overlay, [poly], colors[i])
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

 
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", stream=True, verbose=False)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            if cls in class_map and conf > 0.35:
                if box.id is not None:
                    track_id = int(box.id[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    veh_type = class_map[cls]
                    
                    
                    pt = np.array([[[cx, cy]]], dtype=np.float32)
                    warped_pt = cv2.perspectiveTransform(pt, warp_matrix)[0][0]
                    wx, wy = warped_pt[0], warped_pt[1]
                    
               
                    if track_id not in vehicle_history:
                        vehicle_history[track_id] = []
                        vehicle_warped_history[track_id] = []
                        
                    vehicle_history[track_id].append((cx, cy))
                    vehicle_warped_history[track_id].append((wx, wy))
                    
                    if len(vehicle_history[track_id]) > 30:
                        vehicle_history[track_id].pop(0)
                        vehicle_warped_history[track_id].pop(0)
                        
                
                    if len(vehicle_warped_history[track_id]) >= 15:
                        old_wx, old_wy = vehicle_warped_history[track_id][0]
                        
                      
                        warped_pixel_distance = math.sqrt((wx - old_wx)**2 + (wy - old_wy)**2)
                        time_elapsed = len(vehicle_warped_history[track_id]) / fps
                        
                        speed_ms = (warped_pixel_distance * METER_PER_PIXEL) / time_elapsed
                        vehicle_speeds[track_id] = int(speed_ms * 3.6)

                    if track_id not in vehicle_colors:
                        vehicle_colors[track_id] = (255, 255, 255)

                    for i, poly in enumerate(polygons):
                        if cv2.pointPolygonTest(poly, (cx, cy), False) >= 0:
                            vehicle_colors[track_id] = colors[i] 
                            
                            cv2.polylines(frame, [poly], True, vehicle_colors[track_id], 3) 
                            
                            if track_id not in counted_ids:
                                if len(vehicle_history[track_id]) > 5:
                                    old_cy = vehicle_history[track_id][-5][1]
                                    if cy > old_cy:
                                        lane_counts[i]['in'] += 1
                                    else:
                                        lane_counts[i]['out'] += 1
                                    counted_ids.add(track_id)
                                    class_counts[veh_type] += 1
                            break

                    assigned_color = vehicle_colors[track_id]

                    if len(vehicle_history[track_id]) > 1:
                        pts = np.array(vehicle_history[track_id], np.int32).reshape((-1, 1, 2))
                        cv2.polylines(frame, [pts], False, assigned_color, 3)

                    sub_img = frame[y1:y2, x1:x2]
                    color_rect = np.full(sub_img.shape, assigned_color, dtype=np.uint8)
                    res_img = cv2.addWeighted(sub_img, 0.7, color_rect, 0.3, 0)
                    frame[y1:y2, x1:x2] = res_img

                    cv2.rectangle(frame, (x1, y1), (x2, y2), assigned_color, 2)
                    cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)
                    
                    speed_txt = f"{vehicle_speeds[track_id]} km/h" if track_id in vehicle_speeds else "..."
                    label = f"{veh_type} | {speed_txt}"
                    
                    cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                    cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, assigned_color, 1)

    ui_overlay = frame.copy()
    cv2.rectangle(ui_overlay, (0, 0), (1020, 140), (0, 0, 0), -1)
    cv2.addWeighted(ui_overlay, 0.75, frame, 0.25, 0, frame)
    
    cv2.putText(frame, "Traffic Analysis using YOLOv8s", (20, 25), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
    
    cv2.putText(frame, "LEFT BOUND (Oncoming)", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"L6: IN {lane_counts[5]['in']} | OUT {lane_counts[5]['out']}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[5], 2)
    cv2.putText(frame, f"L5: IN {lane_counts[4]['in']} | OUT {lane_counts[4]['out']}", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[4], 2)
    cv2.putText(frame, f"L4: IN {lane_counts[3]['in']} | OUT {lane_counts[3]['out']}", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[3], 2)

    cv2.putText(frame, "RIGHT BOUND (Outgoing)", (300, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"L1: IN {lane_counts[0]['in']} | OUT {lane_counts[0]['out']}", (300, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[0], 2)
    cv2.putText(frame, f"L2: IN {lane_counts[1]['in']} | OUT {lane_counts[1]['out']}", (300, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[1], 2)
    cv2.putText(frame, f"L3: IN {lane_counts[2]['in']} | OUT {lane_counts[2]['out']}", (300, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[2], 2)
    
    cv2.putText(frame, "VEHICLE BREAKDOWN", (700, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Cars:   {class_counts['Car']}", (700, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Trucks: {class_counts['Truck/Bus']}", (700, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f"TOTAL:  {len(counted_ids)}", (700, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.line(frame, (0, 140), (1020, 140), (0, 255, 255), 2)

    cv2.imshow("Spatial Traffic Analyzer", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()