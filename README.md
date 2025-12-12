# Plant Disease Classification Project
This project focuses on building deep learning models to classify plant leaf diseases using different CNN architectures.
The repository contains experiments, notebooks, and documentation for multiple model architectures.
# 📁 Project Structure
```
.
├── VGG19.ipynb # Training & evaluation using VGG19
├── Resnet.ipynb # Training & evaluation using ResNet
├── mobilenet.ipynb # Training & evaluation using MobileNet
├── train_googlenet.py # Training script for Inception v1 (GoogLeNet)
├── Documentation.pdf # Full project report & documentation
```
# 🌱 Overview
The goal of the project is to detect plant leaf diseases using image classification.
We experiment with several state-of-the-art architectures:
- VGG19
- ResNet
- MobileNet
- GoogLeNet (Inception v1)
Each model is trained and evaluated to compare accuracy, loss, and performance.
# 🚀 How to Run
1. Clone the repository
git clone https://github.com/sama-sameh/plant-disease-project.git
cd plant-disease-project
2. Open any model notebook
You can run:
- VGG19.ipynb
- Resnet.ipynb
- mobilenet.ipynb
in Jupyter Notebook or Google Colab.
3. Run GoogLeNet (Python script)
python train_googlenet.py
# 📊 Models Included
| Model     | File               | Description                          |
| --------- | ------------------ | ------------------------------------ |
| VGG19     | VGG19.ipynb        | Classic deep CNN architecture        |
| ResNet    | Resnet.ipynb       | Residual learning, deeper model      |
| MobileNet | mobilenet.ipynb    | Lightweight model for mobile devices |
| GoogLeNet | train_googlenet.py | Inception v1 implementation          |
# 📄 Documentation
You can find the full methodology, dataset details, results, and analysis in:
📘 Documentation.pdf

