<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
</head>

<body>

<h1>Semiconductor Defect Detection using Deep Learning</h1>

<div class="section">
    <h2>📌 Project Overview</h2>
    <p>
        This project focuses on automatic detection and classification of defects in semiconductor wafer images
        using deep learning techniques. The model is trained using a Convolutional Neural Network (CNN) based on
        pretrained architectures from PyTorch.
    </p>
    <p>
        The system supports:
    </p>
    <ul>
        <li>Training a deep learning model</li>
        <li>Testing on single or multiple images</li>
        <li>Recursive prediction from multiple folders</li>
        <li>ONNX model export for deployment</li>
    </ul>
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
├── dataset/
│   ├── train/
│   │   ├── Bridge/
│   │   ├── Clean/
│   │   └── ...
│   └── test/
│       ├── Bridge/
│       ├── Clean/
│       └── ...
│
├── model/
│   └── best_model.pth
│
├── train.py
├── test_predict_recursive.py
├── test_single.py
├── export_onnx.py
├── requirements.txt
└── README.html
    </pre>
</div>

<div class="section">
    <h2>⚙️ Requirements</h2>
    <p>Install dependencies using:</p>
    <pre>
pip install -r requirements.txt
    </pre>

    <p>Required libraries:</p>
    <ul>
        <li>Python 3.9+</li>
        <li>PyTorch</li>
        <li>Torchvision</li>
        <li>NumPy</li>
        <li>Pillow</li>
        <li>ONNX & ONNX Runtime</li>
        <li>Scikit-learn</li>
        <li>Matplotlib</li>
    </ul>
</div>

<div class="section">
    <h2>🚀 How to Run the Project</h2>

    <h3>1️⃣ Train the Model</h3>
    <pre>
python train.py
    </pre>
    <p>
        This will train the model on the dataset and save the best model in the <code>model/</code> directory.
    </p>

    <h3>2️⃣ Test a Single Image</h3>
    <pre>
python test_single.py --image path/to/image.jpg
    </pre>
    <p>
        Outputs only the predicted defect class for the given image.
    </p>

    <h3>3️⃣ Predict Multiple Images from Multiple Folders</h3>
    <pre>
python test_predict_recursive.py --input test_images/
    </pre>
    <p>
        Recursively scans all subfolders and prints predictions for each image.
    </p>

    <h3>4️⃣ Export Model to ONNX</h3>
    <pre>
python export_onnx.py
    </pre>
    <p>
        Converts the trained PyTorch model into ONNX format for deployment.
    </p>
</div>

<div class="section">
    <h2>📊 Evaluation</h2>
    <p>
        Model performance can be evaluated using accuracy, precision, recall, and confusion matrix
        generated during testing.
    </p>
</div>

<div class="section">
    <h2>🔮 Future Improvements</h2>
    <ul>
        <li>Increase dataset size for better accuracy</li>
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
