# # app.py
# import streamlit as st
# import torch
# import torch.nn as nn
# from torchvision import models, transforms
# from PIL import Image


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# num_classes = 4  # Same as training

# # Define model
# model = models.resnet18(pretrained=False)
# model.fc = nn.Sequential(
#     nn.Linear(model.fc.in_features, 256),
#     nn.ReLU(),
#     nn.Dropout(0.5),
#     nn.Linear(256, num_classes)
# )


# model.load_state_dict(torch.load("model/resnet18_cancer_model.pth", map_location=device))
# model = model.to(device)
# model.eval()


# classes = ["Benign", "Early", "Pre", "Pro"]

# # Image transforms 
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],
#                          [0.229, 0.224, 0.225])
# ])

# #predict
# def predict(image):
#     image = transform(image).unsqueeze(0).to(device)
#     with torch.no_grad():
#         outputs = model(image)
#         _, pred = torch.max(outputs, 1)
#     return classes[pred.item()]

# #app
# st.title("Acute-Lymphoblastic-Leukemia-Classifier ")
# st.write("Upload an image to classify the cancer type.")

# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file).convert("RGB")
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     st.write("Classifying...")
#     prediction = predict(image)
#     st.success(f"Predicted Class: **{prediction}**")

# app.py
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import LayerGradCam

# -------------------------
# 1. Setup
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 4  # Same as training

# Define model (same structure as training)
model = models.resnet18(pretrained=False)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, num_classes)
)

# Load trained weights
model.load_state_dict(torch.load("model/resnet18_cancer_model.pth", map_location=device))
model = model.to(device)
model.eval()

# Class labels
classes = ["Benign", "Early", "Pre", "Pro"]

# Image transforms 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------------
# 2. Grad-CAM setup
# -------------------------
target_layer = model.layer4[1].conv2
gradcam = LayerGradCam(model, target_layer)

def apply_gradcam(image, label_idx=None):
    """Generate Grad-CAM heatmap for given image tensor."""
    image = image.unsqueeze(0).to(device)

    # Forward pass
    output = model(image)
    pred_class = output.argmax(dim=1).item()
    label_idx = label_idx if label_idx is not None else pred_class

    # Grad-CAM attribution
    attributions = gradcam.attribute(image, target=label_idx)
    upsampled_attr = torch.nn.functional.interpolate(
        attributions, size=(224, 224), mode='bilinear'
    )
    heatmap = upsampled_attr.squeeze().cpu().detach().numpy()

    # Normalize heatmap
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    # Convert image back to displayable format
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = image.squeeze().cpu().permute(1, 2, 0).numpy()
    img = std * img + mean
    img = np.clip(img, 0, 1)

    # Overlay heatmap
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.imshow(heatmap, cmap="jet", alpha=0.5)
    ax.axis("off")
    return fig, pred_class

# -------------------------
# 3. Streamlit App
# -------------------------
st.title("Acute Lymphoblastic Leukemia Classifier with Explainable AI")
st.write("Upload an image to classify the cancer type and view an explanation (Grad-CAM).")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("Classifying and explaining...")

    # Transform image
    transform_img = transform(image)

    # Apply Grad-CAM
    fig, pred_class = apply_gradcam(transform_img)

    # Display prediction
    st.success(f"Predicted Class: **{classes[pred_class]}**")

    # Display Grad-CAM heatmap
    st.pyplot(fig)
    st.caption("Highlighted regions show which parts of the image influenced the prediction most.")

