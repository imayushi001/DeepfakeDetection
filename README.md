Deepfake Detection (Gradio App)
A simple Gradio application for detecting deepfakes in face images, providing class probabilities (real vs fake), Grad-CAM explainability overlays, and basic evaluation plots (confusion matrix and ROC curve) generated from bundled examples.

Features
Face detection using MTCNN from facenet-pytorch.
Classifier based on InceptionResnetV1 fine-tuned for binary output.
Explainability via Grad-CAM heatmaps over the detected face.
Evaluation artifacts: confusion matrix and ROC curve built from sample images.
Interactive UI using Gradio.
Repository structure
app.py: Gradio app and inference/evaluation logic
requirements.txt: Python dependencies
resnetinceptionv1_epoch_32.pth: trained model checkpoint
examples/: sample images (some real and fake frames)
confusion_matrix.png, roc_curve.png: generated on startup/evaluation
Requirements
Python 3.8+
A working PyTorch installation (CPU works; CUDA is used if available)
# Deepfake Detection

pip install -r requirements.txt
Note: If you need GPU support, install the CUDA-enabled version of PyTorch per the official instructions before running the app.


Usage
Render a Grad-CAM overlay highlighting influential regions
Display the confusion matrix and ROC curve images generated from sample data
Notes about examples and startup
On startup, the app evaluates the model on a small set of bundled images in examples/ and saves confusion_matrix.png and roc_curve.png.
The current code expects an examples.zip at the repo root and attempts to unzip it. If you already have an examples/ folder (as in this repo), you can:
Provide examples.zip yourself, or
Comment out or remove the unzip block in app.py:
# with zipfile.ZipFile("examples.zip", "r") as zip_ref:
#     zip_ref.extractall(".")
Model details and thresholds
Backbone: InceptionResnetV1 (facenet-pytorch, pretrained on vggface2), adapted for binary classification with a single logit.
Output is passed through a sigmoid; the app currently uses a decision threshold of 0.7, mapping to:
score > 0.7 → real
score ≤ 0.7 → fake You can adjust this threshold in app.py to tune precision/recall.
Known limitations
The example-driven evaluation (confusion matrix and ROC) is based on a tiny, bundled set and is not representative of real-world performance.
Face detection failures (no detectable face) will raise an error for that input.
The model file resnetinceptionv1_epoch_32.pth must be present at the repo root.
Troubleshooting
"No face detected": Ensure the uploaded image contains a clear, frontal face; try higher-resolution images.
CUDA-related errors: Ensure your PyTorch installation matches your CUDA toolkit, or run on CPU by uninstalling CUDA-enabled PyTorch.
Missing examples on startup: Either add examples.zip or comment out the unzip step as noted above.
Port conflicts: If Gradio fails to launch, specify a port, e.g. gradio.Interface(...).launch(server_port=7861).
grad-cam for visualization utilities
Gradio for the web UI