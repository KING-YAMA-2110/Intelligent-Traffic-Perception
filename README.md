# 🚦 Intelligent Traffic Perception (ITP)

## 🏆 KrackHack 2.0

Welcome to our **Intelligent Traffic Perception (ITP)** project! This repository contains all necessary code, datasets, and model implementations for **real-time object detection, drivable area segmentation, and lane line detection** using **HybridNets and YOLO**. The project aims to enhance **traffic perception** for autonomous vehicles, smart surveillance, and intelligent transportation systems.

---

## 📌 Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Dataset](#dataset)
- [Important Links](#important_links)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Contributors](#contributors)

---

## 🚀 Introduction

The **Intelligent Traffic Perception (ITP)** system integrates deep learning models to analyze road environments efficiently. The primary objectives include:

- **Traffic Object Detection** 🚗 (cars, pedestrians, traffic signals, etc.)
- **Drivable Area Segmentation** 🛣 (detecting the safe areas to drive)
- **Lane Line Detection** 🛤 (identifying lane markings on roads)
- **Fusion of YOLO & HybridNets** 🔄 (combining multiple models for accuracy)
- **Real-time Video Processing** 🎥 (detecting objects in live footage)

---

## ✨ Features

✔ **Pretrained YOLO & HybridNets** for object detection and segmentation  
✔ **Supports real-time video processing** for intelligent surveillance  
✔ **Fusion-based confidence strategy** for better decision-making  
✔ **Automated dataset extraction and preprocessing**  

---

## 📂 Dataset

We use **BDD10K** and custom datasets with annotated JSON files containing bounding boxes for objects. The dataset is structured as:

```
📁 extracted_data/
 ├── train/
 │   ├── img/  (Training images)
 │   ├── ann/  (Annotations in JSON format)
 ├── val/
 │   ├── img/
 │   ├── ann/
 ├── test/
 │   ├── img/
 │   ├── ann/

```
---
## 🔗 Important Links

- **Model Weights:** [Google Drive](https://drive.google.com/drive/folders/1hWp71RspIPoxh7CizgcR7nB8GKfi_WB8?usp=sharing)  
- **Repository Explanation Video:** [YouTube](https://youtu.be/v2BzX-Ri7Ww)  

---

## 🛠 Installation

Clone this repository:
```
bash
git clone https://github.com/yourusername/ITP-Hackathon.git
cd ITP-Hackathon
```

Install dependencies:
```
bash
pip install -r requirements.txt
```

## 🚀 Usage

### **1️⃣ Running Object Detection on Images**
```
python
from ultralytics import YOLO
import cv2

yolo = YOLO("yolo11s.pt")  # Load YOLO model
image_path = "test.jpg"
img = cv2.imread(image_path)
results = yolo(img)
```

### **2️⃣ Running ResNet Feature Extraction**
```
python
import torch
import torchvision.models as models

resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-2])
resnet.eval()
```

### **3️⃣ Running HybridNets for Segmentation**
```
python
import torch
model = torch.hub.load('datvuthanh/hybridnets', 'hybridnets', pretrained=True)
model.eval()
```

## 🧠 Model Details

### **1️⃣ YOLO Object Detection**

YOLO detects objects like cars, pedestrians, and traffic signs with high accuracy.

```
🔲🔲🔲 [🚗] 🔲🔲 [🚶] 🔲🔲🔲
  ↳ Bounding Boxes around objects
```

### **2️⃣ HybridNets Segmentation**

It segments drivable areas and lane lines for intelligent navigation.

```
🛣🛣🛣🛣  ➝ [Drivable Area]
  🚗  🚗  ➝ [Lane Lines]
```

### **3️⃣ Fusion of YOLO & HybridNets**

We use a confidence-based fusion strategy to merge predictions from both models for better accuracy.

---

## 👥 Contributors

- **Abhijeet Singh** 
- **Alok Kumar** 
- **Abhiraj Kuntal** 
- **Shubhankit Singh**
  
For queries, contact **kingyama2110@gmail.com**.

---

