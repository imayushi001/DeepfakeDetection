import gradio as gr
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
import os
import numpy as np
from PIL import Image
import zipfile
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Unzip examples
with zipfile.ZipFile("examples.zip", "r") as zip_ref:
    zip_ref.extractall(".")

# Set device
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Initialize MTCNN and the InceptionResnetV1 model
mtcnn = MTCNN(
    select_largest=False,
    post_process=False,
    device=DEVICE
).to(DEVICE).eval()

model = InceptionResnetV1(
    pretrained="vggface2",
    classify=True,
    num_classes=1,
    device=DEVICE
)

# Load the model checkpoint
checkpoint = torch.load("resnetinceptionv1_epoch_32.pth", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()

# Load examples
EXAMPLES_FOLDER = 'examples'
examples_names = os.listdir(EXAMPLES_FOLDER)
examples = []
for example_name in examples_names:
    example_path = os.path.join(EXAMPLES_FOLDER, example_name)
    label = 1 if example_name.startswith('real') else 0  # 1 for real, 0 for fake
    example = {
        'path': example_path,
        'label': label
    }
    examples.append(example)
np.random.shuffle(examples)  # Shuffle examples

def plot_confusion_matrix(cm):
    """Plot and save the confusion matrix."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()

def plot_roc_curve(fpr, tpr, auc):
    """Plot and save the ROC curve."""
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()

def evaluate_model():
    """Evaluate the model and generate confusion matrix and ROC curve."""
    all_labels = []
    all_predictions = []
    
    for example in examples:
        img_path = example['path']
        true_label = example['label']
        input_image = Image.open(img_path)

        face = mtcnn(input_image)
        if face is None:
            continue
        
        face = face.unsqueeze(0)
        face = F.interpolate(face, size=(256, 256), mode='bilinear', align_corners=False)
        face = face.to(DEVICE)
        face = face / 255.0

        with torch.no_grad():
            output = torch.sigmoid(model(face).squeeze(0))
            prediction = 0 if output.item() > 0.7 else 1

        all_labels.append(true_label)
        all_predictions.append(prediction)

    cm = confusion_matrix(all_labels, all_predictions)
    fpr, tpr, _ = roc_curve(all_labels, all_predictions)
    auc = roc_auc_score(all_labels, all_predictions)

    plot_confusion_matrix(cm)
    plot_roc_curve(fpr, tpr, auc)

    return cm, fpr, tpr, auc

# Evaluate the model and generate plots
cm, fpr, tpr, auc = evaluate_model()

def predict(input_image: Image.Image, true_label: str):
    """Predict the label of the input_image and generate explainability."""
    face = mtcnn(input_image)
    if face is None:
        raise Exception('No face detected')
    face = face.unsqueeze(0)  # Add the batch dimension
    face = F.interpolate(face, size=(256, 256), mode='bilinear', align_corners=False)

    prev_face = face.squeeze(0).permute(1, 2, 0).cpu().detach().int().numpy()
    prev_face = prev_face.astype('uint8')

    face = face.to(DEVICE)
    face = face.to(torch.float32)
    face = face / 255.0
    face_image_to_plot = face.squeeze(0).permute(1, 2, 0).cpu().detach().int().numpy()

    target_layers = [model.block8.branch1[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(0)]

    grayscale_cam = cam(input_tensor=face, targets=targets, eigen_smooth=True)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(face_image_to_plot, grayscale_cam, use_rgb=True)
    face_with_mask = cv2.addWeighted(prev_face, 1, visualization, 0.5, 0)

    with torch.no_grad():
        output = torch.sigmoid(model(face).squeeze(0))
        prediction = "real" if output.item() > 0.7 else "fake"

        real_prediction = output.item()
        fake_prediction = 1 - output.item()

        confidences = {
            'real': real_prediction,
            'fake': fake_prediction
        }
    
    return confidences, face_with_mask, 'confusion_matrix.png', 'roc_curve.png'

# Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.components.Image(label="Input Image", type="pil"),
        gr.components.Text(label="Your Text Input")
    ],
    outputs=[
        gr.components.Label(label="Class"),
        gr.components.Image(label="Face with Explainability", type="numpy"),
        gr.components.Image(label="Confusion Matrix", type="filepath"),
        gr.components.Image(label="ROC Curve", type="filepath")
    ],
    examples=[[examples[i]["path"], examples[i]["label"]] for i in range(10)],
    cache_examples=True
).launch()
