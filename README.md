Deepfake Detection (Gradio App) 🎭


A simple Gradio application for detecting deepfakes in face images.
The app provides:
✅ Class probabilities (real vs fake)
✅ Grad-CAM heatmaps for explainability
✅ Evaluation plots (Confusion Matrix & ROC Curve)

✨ Features

🔍 Face Detection using MTCNN

🧠 Binary Classifier (InceptionResnetV1, fine-tuned for deepfake detection)

🎨 Explainability via Grad-CAM overlays

📊 Evaluation Artifacts: confusion matrix + ROC curve

🌐 Interactive Web UI powered by Gradio

📂 Repository Structure
DeepfakeDetection/
├── app.py                       # Main Gradio app (inference + evaluation logic)
├── requirements.txt             # Python dependencies
├── resnetinceptionv1_epoch_32.pth  # Trained model checkpoint
├── examples/                    # Sample images (real & fake)
├── confusion_matrix.png         # Evaluation confusion matrix
├── roc_curve.png                # Evaluation ROC curve

⚙️ Installation

Clone the repository:

git clone https://github.com/<your-username>/DeepfakeDetection.git
cd DeepfakeDetection


Install dependencies:

pip install -r requirements.txt


👉 For GPU support, install the CUDA-enabled version of PyTorch per official instructions
.

🚀 Running the App

Run the following command:

python app.py


This will launch a Gradio interface and print a local URL (and optionally a public share link).
Open it in your browser to interact with the app.

🖼️ Usage

Open the Gradio URL shown in the terminal.

Upload a face image (or select from bundled examples).

The app will:

Detect the largest face (via MTCNN)

Classify the face → Real or Fake (with probabilities)

Render a Grad-CAM heatmap

Display confusion matrix & ROC curve

📊 Notes on Examples

On startup, the app evaluates the model on the bundled dataset (examples/) and generates:

confusion_matrix.png

roc_curve.png

The code expects an examples.zip file. If you already have an examples/ folder:

Provide examples.zip, or

Comment/remove the unzip block in app.py:

# with zipfile.ZipFile("examples.zip", "r") as zip_ref:
#     zip_ref.extractall(".")

🧠 Model Details

Backbone: InceptionResnetV1 (facenet-pytorch, pretrained on VGGFace2)

Modified for binary classification with a single logit

Output passed through sigmoid

Decision threshold: 0.7

score > 0.7 → Real

score ≤ 0.7 → Fake

👉 Threshold can be tuned in app.py for precision/recall trade-offs.

⚠️ Known Limitations

Bundled dataset is tiny → not representative of real-world accuracy

Face detection failures → app raises error

Model file resnetinceptionv1_epoch_32.pth must exist at repo root

🛠️ Troubleshooting

❌ "No face detected" → use clear, frontal, high-resolution images

❌ CUDA errors → check PyTorch + CUDA compatibility, or run on CPU

❌ Examples missing → add examples.zip or skip unzip logic

❌ Port conflict → launch on a custom port:

interface.launch(server_port=7861)

🙏 Acknowledgments

facenet-pytorch
 – MTCNN & InceptionResnetV1

grad-cam
 – Visualization utilities

Gradio
 – Interactive web UI
