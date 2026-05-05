# 👁️ The Vision AI — An Eye for the Blind

## 📌 Overview

The Vision AI is an assistive application designed to help visually impaired individuals interact with their surroundings using computer vision and audio feedback.

It provides:

* 🎯 Real-time object detection
* 🧾 Text recognition (OCR)
* 🔊 Voice-based feedback

---

## 🚀 Features

* 🎥 Live camera feed with high FPS
* 🎯 Object detection using YOLOv3-tiny
* 🧾 OCR-based text reading using Tesseract
* 🔊 Real-time speech output (Text-to-Speech)
* 🧵 Multi-threaded architecture for smooth performance
* 📝 Automatic logging system
* 🖥️ Simple and accessible UI

---

## 🛠️ Tech Stack

* **Language:** Python
* **Libraries:**

  * OpenCV (`cv2`)
  * cvlib (YOLO-based detection)
  * pytesseract (OCR)
  * Tkinter (GUI)
  * PIL (Image processing)
  * NumPy

---

## 📂 Project Structure

```
projectvai.py        # Main application file
README.md            # Project documentation
VisionAI_Detection_Log.txt  # Logs (auto-generated)
```

---

## ⚙️ Installation

### 1. Clone the repository

```
git clone https://github.com/your-username/vision-ai.git
cd vision-ai
```

### 2. Install dependencies

```
pip install opencv-python cvlib pytesseract pillow numpy
```

### 3. Install Tesseract OCR

* Download from: https://github.com/tesseract-ocr/tesseract
* Add to system PATH

---

## ▶️ Usage

Run the application:

```
python projectvai.py
```

### Modes:

* 🎯 Object Detection Mode
* 🗣 Text Reader Mode
* 🛑 Stop Mode
* ❌ Exit

---

## 🧠 How It Works

### Object Detection

* Uses YOLOv3-tiny model via cvlib
* Detects common objects in real time
* Announces detected objects via speech

### Text Recognition

* Uses Tesseract OCR
* Extracts text from camera frames
* Reads aloud detected text


---

## 🔮 Future Enhancements

* 🌐 Multi-language support
* 📱 Mobile app version
* 🧠 Custom AI models (TensorFlow/PyTorch)
* 📍 Navigation assistance
* ☁️ Cloud-based processing

---

## 🤝 Contributing

Contributions are welcome!
Feel free to fork this repo and submit a pull request.

---

## 📜 License

This project is intended for educational and demonstration purposes only.
You may not use, copy, modify, or distribute this project without explicit permission from the author.

---

## 👩‍💻 Author

Rakshitha
AI/ML Enthusiast & Computer Science Student

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
