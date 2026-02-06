<!DOCTYPE html>
<html lang="en">
<body>

<h1>🧠 Semiconductor Defect Detection using Deep Learning (WaferVision-Ai)</h1>

<div class="section">
    <h2>📌 Project Overview</h2>
    <p>
        This project implements an AI-based automated system for detecting and classifying
        surface defects in semiconductor wafer images using deep learning and computer vision.
        The goal is to replace manual and rule-based inspection methods with a fast, accurate,
        and scalable solution.
    </p>
</div>

<div class="section">
    <h2>🧠 Defect Classes</h2>
    <ul>
        <li>Bridge</li>
        <li>Clean</li>
        <li>Complex</li>
        <li>Cracks</li>
        <li>Flat</li>
        <li>Foreign Material</li>
        <li>Line Edge Roughness</li>
        <li>Linear</li>
        <li>Scratches</li>
        <li>Z-axis</li>
    </ul>
</div>

<div class="section">
    <h2>📁 Project Structure</h2>
    <pre>
Semiconductor_Defect_Detection/
│
├── data/
│   └── train/
│       ├── Bridge/
│       ├── Clean/
│       └── ...
│
├── test/
│   └── Testimg.png
│
├── model/
│   ├── defect_model.pth
│   ├── defect_model.onnx
│   ├── defect_model.onnx.data
│   └── class_mapping.json
│
├── train.py
├── test.py
├── test_images.py
├── requirements.txt
└── README.html
    </pre>
</div>

<div class="section">
    <h2>⚙️ Requirements</h2>
    <p>Install dependencies using:</p>
    <pre>pip install -r requirements.txt</pre>

    Requirements :
    1. Python 3.9+
    2. PyTorch & Torchvision
    3. NumPy
    4. Pillow
    5. OpenCV
    6. ONNX & ONNX Runtime
    7. Scikit-learn
    8. Matplotlib
    
</div>

<div class="section">
    <h2>🚀 How to Run the Project</h2>

    1️⃣ Train the Model
    python train.py
    Trains the CNN model and saves it in the model directory.

    2️⃣ Test a Single Image
    python test.py 
</div>

<div class="section">
    <h2>🔗 Resources & References</h2>
    <p>
        The following datasets were used for building and validating
        the semiconductor defect detection system:
    </p>
            <a href="https://www.kaggle.com/datasets" target="_blank" style="color:#38bdf8;">
                Semiconductor Wafer Defect Dataset
            </a>
       
</div>


<div class="section">
    <h2>📊 Evaluation</h2>
    <p>
        Model performance is evaluated using accuracy, precision, recall, and confusion matrix
        during the testing phase.
    </p>
</div>

<div class="section">
    <h2>💡 Innovation & Highlights</h2>
    <ul>
        <li>Automated semiconductor wafer inspection</li>
        <li>Deep learning–based multi-class defect classification</li>
        <li>Eliminates manual inspection dependency</li>
        <li>Deployment-ready using ONNX</li>
    </ul>
</div>

<div class="section">
    <h2>🔮 Future Improvements</h2>
    <ul>
        <li>Increase dataset size</li>
        <li>Use advanced architectures (EfficientNet, ViT)</li>
        <li>Real-time inference pipeline</li>
        <li>Web or API-based deployment</li>
    </ul>
</div>

<div class="section">
    <h2>👨‍💻 Author</h2>
    <p>
        <strong>Om Nimmalwar</strong><br>
        MCA Student | Data Scientist & AI Enthusiast
    </p>
</div>

</body>
</html>
