import streamlit as st
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
import dlib
import numpy as np

# Set Streamlit page config
st.set_page_config(page_title="Emotion Detector", layout="centered")

# Initialize dlib face detector
detector = dlib.get_frontal_face_detector()

# Define Emotion Classes
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Define CNN Model
class SimpleCNN(nn.Module):
    def __init__(self, output):
        super(SimpleCNN, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, padding=0, stride=1),     # (112x112x1) -> (108x108x8)
            nn.AvgPool2d(kernel_size=3, stride=3),                   # (108x108x8) -> (36x36x8)
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0),    # (36x36x8) -> (34x34x16)
            nn.AvgPool2d(kernel_size=2, stride=2),                   # (34x34x16) -> (17x17x16)
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(17*17*16, 128),
            nn.Linear(128, 64),
            nn.Linear(64, output)
        )

    def forward(self, x):
        x = self.feature(x)
        x = F.relu(x)
        x = self.classifier(x)
        return x
class SimpleCNN1(nn.Module):
    def __init__(self, output):
        super(SimpleCNN1, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, padding=0, stride=1)  # 224x224x1 -> 220x220x8
        self.pool1 = nn.AvgPool2d(kernel_size=3, stride=3)               # -> 73x73
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0)# -> 71x71x16
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)               # -> 35x35x16
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(35 * 35 * 16, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output)

    def forward(self, X):
        X = self.pool1(self.conv1(X))
        X = self.pool2(self.conv2(X))
        X = self.flatten(X)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        return self.fc3(X)
        
# Load trained models
@st.cache_resource
def load_models():
    model = SimpleCNN(output=7)
    model.load_state_dict(torch.load("emotion_data_final (1).pth", map_location=torch.device('cpu')))
    model.eval()

    model1 = SimpleCNN1(output=7)
    model1.load_state_dict(torch.load("emotion_data_final (1).pth", map_location=torch.device('cpu')))
    model1.eval()
    
    return model, model1

model, model1 = load_models()

# Define transformation
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

transform1 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Function to detect and crop face using dlib
def crop_face(img_pil):
    img_np = np.array(img_pil.convert("RGB"))
    dets = detector(img_np, 1)
    if len(dets) == 0:
        return None
    face = dets[0]
    cropped = img_np[face.top():face.bottom(), face.left():face.right()]
    if cropped.shape[0] == 0 or cropped.shape[1] == 0:
        return None
    resized = cv2.resize(cropped, (112, 112))
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    return Image.fromarray(gray)

# UI
st.title("😊 Emotion Detection from Image")
st.write("Upload an image with a visible face, and I’ll predict the emotion!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file)
        face_img = crop_face(img)
        if face_img is None:
            st.warning("No face detected in the image.")
        else:
            image = transform(face_img).unsqueeze(0)  # Add batch dimension
            image1 = transform1(face_img).unsqueeze(0)
            
            # Predict with Model 1
            with torch.no_grad():
                output = model(image)
                probs = torch.softmax(output, dim=1)
                top_probs, top_indices = torch.topk(probs, k=3)

                output1 = model1(image1)
                probs1 = torch.softmax(output1, dim=1)
                top_probs1, top_indices1 = torch.topk(probs1, k=3)

            # Display
            col1, col2 = st.columns(2)

            with col1:
                st.image(uploaded_file, caption="Original Image", width=250)
            
            with col2:
                st.subheader("Model 1 Prediction")
                st.write(f"Top Emotion: **{emotions[top_indices[0][0]]}**")
                st.write("Top 3 Probabilities:")
                for i in range(3):
                    st.write(f"{emotions[top_indices[0][i]]}: {top_probs[0][i].item():.4f}")

                st.markdown("---")
                st.subheader("Model 2 Prediction")
                st.write(f"Top Emotion: **{emotions[top_indices1[0][0]]}**")
                st.write("Top 3 Probabilities:")
                for i in range(3):
                    st.write(f"{emotions[top_indices1[0][i]]}: {top_probs1[0][i].item():.4f}")

    except Exception as e:
        st.error(f"Error: {e}")
