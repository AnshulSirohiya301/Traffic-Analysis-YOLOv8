# 🚦 Traffic Analysis using YOLOv8s

An enterprise-grade, 6-lane bi-directional traffic analytics pipeline powered by Computer Vision. This project utilizes YOLOv8 for vehicle tracking and classification, combined with OpenCV perspective transformation (Homography) to accurately estimate real-world vehicle speeds without camera distortion.

## ✨ Key Features
* **Multi-Lane Spatial Mapping:** Tracks vehicles across 6 independent lanes (Incoming & Outgoing).
* **Bird's-Eye View Speed Estimation:** Uses `cv2.getPerspectiveTransform` to flatten the angled camera feed into a 2D top-down grid, mathematically eliminating depth distortion for stable `km/h` calculations.
* **Vehicle Classification:** Categorizes traffic flow into Cars, Trucks/Buses, and Motorcycles.
* **State-Managed UI:** Features a dynamic "Chameleon" UI where vehicle bounding boxes and trailing trace-lines instantly adapt to match their specific lane color.

## 🛠️ Tech Stack
* **AI Model:** Ultralytics YOLOv8s (with ByteTrack)
* **Computer Vision:** OpenCV (cv2)
* **Matrix Operations:** NumPy
* **Language:** Python 3.x

## 🚀 Getting Started

### 1. Install Dependencies
Clone the repository and install the required libraries:
```bash
git clone [https://github.com/yourusername/Traffic-Analysis-YOLOv8.git](https://github.com/yourusername/Traffic-Analysis-YOLOv8.git)
cd Traffic-Analysis-YOLOv8
pip install -r requirements.txt