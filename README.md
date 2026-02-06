<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Semiconductor Defect Detection using Deep Learning</title>
    <style>
        body {
            font-family: Arial, Helvetica, sans-serif;
            background-color: #0b1220;
            color: #e5e7eb;
            line-height: 1.6;
            padding: 30px;
        }
        h1, h2, h3 {
            color: #38bdf8;
        }
        ul {
            margin-left: 20px;
        }
        pre {
            background: #020617;
            padding: 15px;
            border-radius: 6px;
            overflow-x: auto;
            color: #a5f3fc;
        }
        .section {
            margin-bottom: 40px;
        }
        .highlight {
            color: #22c55e;
        }
    </style>
</head>

<body>

<h1>🧠 Semiconductor Defect Detection using Deep Learning</h1>

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

    <ul>
        <li>Python 3.9+</li>
        <li>PyTorch & Torchvision</li>
        <li>NumPy</li>
        <li>Pillow</li>
        <li>OpenCV</li>
        <li>ONNX & ONNX Runtime</li>
        <li>Scikit-learn</li>
        <li>Matplotlib</li>
    </ul>
</div>

<div class="section">
    <h2>🚀 How to Run the Project</h2>

    <h3>1️⃣ Train the Model</h3>
    <pre>python train.py</pre>
    <p>Trains the CNN model and saves it in the <span class="highlight">model/</span> directory.</p>

    <h3>2️⃣ Test a Single Image</h3>
    <pre>python test.py --image path/to/image.jpg</pre>

    <h3>3️⃣ Predict Multiple Images (Recursive)</h3>
    <pre>python test_images.py --input test_images/</pre>

    <h3>4️⃣ Export Model to ONNX</h3>
    <pre>python export_onnx.py</pre>
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
