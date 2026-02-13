# Semantic-Segmentation-using-U-Net-Real-time-Object-Detection-with-YOLOv8
Hand-on computer vision practical on semantic segmentation of brain tumor image using U-Net and real-time object detection on traffic video using YOLOv8.
The focus is on understanding model pipelines, training flow, and inference rather than only theory.

---

## Tech Stack

- Python  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  
- OpenCV  
- Ultralytics YOLOv8  

---

## Practical 3 – Semantic Segmentation with U-Net

### Objective
Build, train, and evaluate a U-Net model for **pixel-level binary segmentation**.

### Dataset
The notebook demonstrates the pipeline using **synthetic placeholder data**.  
It is structured so it can be replaced with real datasets such as medical images (e.g., ultrasound or polyp segmentation).

### This practical includes

- Implementation of U-Net from scratch  
- Encoder–decoder design with skip connections  
- Double convolution blocks with Batch Normalization  
- Downsampling via MaxPooling  
- Upsampling via Transposed Convolutions  
- Dice loss for segmentation  
- Metrics: Accuracy and Mean IoU  
- Dummy data generation for demonstration  
- Model training  
- Visualization of:
  - input image  
  - ground truth mask  
  - predicted mask  

### Model pipeline (simplified)

Input → Encoder → Bottleneck → Decoder with Skip Connections → Sigmoid Mask


---

## Practical 4 – Real-time Object Detection with YOLOv8

### Objective
Run modern YOLOv8 for real-time object detection on webcam or video.

### Model
Pre-trained **yolov8n.pt** (nano variant) from Ultralytics.

### This practical includes

- Loading a pre-trained YOLOv8 model  
- Running inference on live webcam feed or video  
- Using built-in visualization to draw bounding boxes  
- Real-time display with OpenCV  
- Option to use a custom trained model  
- Option to run on single images or save outputs  

### Detection pipeline (simplified)

Frame → YOLOv8 Inference → Bounding Boxes → Display


---

## Learning Outcomes

Through these practicals, I learned:

- Building segmentation networks from scratch  
- Understanding encoder–decoder structures  
- Working with Dice loss & IoU  
- Preparing pipelines for real datasets  
- Using state-of-the-art detection models  
- Running real-time inference  



