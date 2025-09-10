# import streamlit as st
# import torch
# import torch.nn as nn
# from torchvision import models, transforms
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
# from captum.attr import LayerGradCam
# import io


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# num_classes = 4  # Same as training

# # Define model (same structure as training)
# model = models.resnet18(pretrained=False)
# model.fc = nn.Sequential(
#     nn.Linear(model.fc.in_features, 256),
#     nn.ReLU(),
#     nn.Dropout(0.5),
#     nn.Linear(256, num_classes)
# )

# # Load trained weights
# model.load_state_dict(torch.load("model/resnet18_cancer_model.pth", map_location=device))
# model = model.to(device)
# model.eval()

# # Class labels
# classes = ["Benign", "Early", "Pre", "Pro"]

# # Image transforms 
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],
#                          [0.229, 0.224, 0.225])
# ])


# # Grad-CAM setup

# target_layer = model.layer4[1].conv2
# gradcam = LayerGradCam(model, target_layer)

# def apply_gradcam(image, label_idx=None):
#     """Generate Grad-CAM heatmap for given image tensor."""
#     image = image.unsqueeze(0).to(device)

#     # Forward pass
#     output = model(image)
#     pred_class = output.argmax(dim=1).item()
#     label_idx = label_idx if label_idx is not None else pred_class

#     # Grad-CAM attribution
#     attributions = gradcam.attribute(image, target=label_idx)
#     upsampled_attr = torch.nn.functional.interpolate(
#         attributions, size=(224, 224), mode='bilinear'
#     )
#     heatmap = upsampled_attr.squeeze().cpu().detach().numpy()

#     # Normalize heatmap
#     heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

#     # Convert image back to displayable format
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     img = image.squeeze().cpu().permute(1, 2, 0).numpy()
#     img = std * img + mean
#     img = np.clip(img, 0, 1)

#     # Overlay heatmap
#     fig, ax = plt.subplots()
#     ax.imshow(img)
#     ax.imshow(heatmap, cmap="jet", alpha=0.5)
#     ax.axis("off")
#     return fig, pred_class


# # Streamlit App

# st.title("Acute Lymphoblastic Leukemia Classifier with Explainable AI")
# st.write("Upload an image to classify the cancer type and view an explanation (Grad-CAM).")

# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file).convert("RGB")
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     st.write("Classifying and explaining...")

#     # Transform image
#     transform_img = transform(image)

#     # Apply Grad-CAM
#     fig, pred_class = apply_gradcam(transform_img)

#     # Display prediction
#     st.success(f"Predicted Class: **{classes[pred_class]}**")

#     # Display Grad-CAM heatmap
#     st.pyplot(fig)
#     st.caption("Highlighted regions show which parts of the image influenced the prediction most.")

  
#     # Download Grad-CAM image
   
#     buf = io.BytesIO()
#     fig.savefig(buf, format="png", bbox_inches='tight')  # tight removes extra margins
#     buf.seek(0)
#     st.download_button(
#         label="Download Grad-CAM Image",
#         data=buf,
#         file_name="gradcam_result.png",
#         mime="image/png"
#     )
import os
import io
import base64
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
from torchvision import models, transforms
from captum.attr import LayerGradCam
from openai import AzureOpenAI


# -------------------------
# 1. Load Azure Keys
# -------------------------
keys_file = r"C:\Users\94718\Downloads\CW\ANN-based-Acute-Lymphoblastic-Leukemia-Classifier\genai\azure_keys.txt"
with open(keys_file, "r") as f:
    lines = f.read().strip().splitlines()
    key_dict = dict(line.split("=", 1) for line in lines if "=" in line)

os.environ["AZURE_OPENAI_API_KEY"] = key_dict.get("AZURE_OPENAI_API_KEY", "")
os.environ["AZURE_OPENAI_ENDPOINT"] = key_dict.get("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_VERSION = key_dict.get("AZURE_OPENAI_API_VERSION", "2024-02-01")
GPT_DEPLOYMENT_NAME = key_dict.get("GPT_DEPLOYMENT_NAME", "gpt-4o")

client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
)


# -------------------------
# 2. Model Setup
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 4
classes = ["Benign", "Early", "Pre", "Pro"]

# Define same model as training
model = models.resnet18(pretrained=False)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, num_classes),
)
model.load_state_dict(torch.load("model/resnet18_cancer_model.pth", map_location=device))
model = model.to(device)
model.eval()

# Preprocessing transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# Grad-CAM setup
target_layer = model.layer4[1].conv2
gradcam = LayerGradCam(model, target_layer)


# -------------------------
# 3. Grad-CAM Function
# -------------------------
def generate_gradcam(image_tensor, label_idx=None):
    image_tensor = image_tensor.unsqueeze(0).to(device)

    # Forward pass
    output = model(image_tensor)
    pred_class = output.argmax(1).item()
    label_idx = label_idx if label_idx is not None else pred_class

    # Grad-CAM attribution
    attr = gradcam.attribute(image_tensor, target=label_idx)
    attr_upsampled = torch.nn.functional.interpolate(
        attr, size=(224, 224), mode="bilinear"
    ).squeeze().cpu().detach().numpy()

    # Normalize
    attr_upsampled = (attr_upsampled - attr_upsampled.min()) / (
        attr_upsampled.max() - attr_upsampled.min()
    )

    # Convert input image back to displayable format
    img_np = image_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = np.clip(img_np * std + mean, 0, 1)

    # Overlay heatmap
    fig, ax = plt.subplots()
    ax.imshow(img_np)
    ax.imshow(attr_upsampled, cmap="jet", alpha=0.5)
    ax.axis("off")

    # Save Grad-CAM to memory buffer
    buf = io.BytesIO()
    fig.savefig(buf, format="PNG", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return pred_class, buf


# -------------------------
# 4. GPT Explanation
# -------------------------
def generate_gpt_explanation(pred_class_idx, gradcam_buf):
    gradcam_base64 = base64.b64encode(gradcam_buf.read()).decode("utf-8")

    prompt = f"""
You are an expert in acute lymphoblastic leukemia (ALL), supporting medical students, lab technicians, and patients in interpreting 
peripheral blood smear (PBS) images.

A ResNet18 model has predicted the class '{classes[pred_class_idx]}' for a PBS image. 
A Grad-CAM heatmap highlights the important image regions influencing this prediction.

Please provide concise and meaningful results (avoid unnecessary long text):

1. **Medical/Technical Explanation** (for medical students and lab technicians):
   - Interpret the ResNet18 prediction with reference to the Grad-CAM heatmap.
   - Describe which regions were important and why.
   - Provide a brief technical evaluation and the clinical significance of these findings.
   - Add short notes relevant for medical and lab technicians.

2. **Patient-Friendly Explanation**:
   - Act as a doctor speaking to a patient.
   - Clearly explain what this prediction means.
   - Suggest possible next steps and treatment plan guidance in simple terms.
"""


    response = client.chat.completions.create(
        model=GPT_DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful AI for cancer image interpretation."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{gradcam_base64}"}},
                ],
            },
        ],
    )
    return response.choices[0].message.content


# -------------------------
# 5. Streamlit UI
# -------------------------
st.title("Acute Lymphoblastic Leukemia Classifier with Explainable AI")
st.write("Upload an image to classify the cancer type and generate an explanation (Grad-CAM + GPT).")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("Classifying and generating explanation...")

    # Transform and run Grad-CAM
    img_tensor = transform(image)
    pred_class_idx, gradcam_buf = generate_gradcam(img_tensor)

    # Show prediction
    st.success(f"Predicted Class: **{classes[pred_class_idx]}**")

    # Display Grad-CAM
    st.image(gradcam_buf, caption="Grad-CAM Heatmap", use_column_width=True)

    # Generate GPT explanation
    explanation = generate_gpt_explanation(pred_class_idx, gradcam_buf)
    st.subheader("Medical & Technical Explanation")
    st.write(explanation)

    # Reset buffer for download
    gradcam_buf.seek(0)
    st.download_button(
        label="Download Grad-CAM Image",
        data=gradcam_buf,
        file_name="gradcam_result.png",
        mime="image/png",
    )

