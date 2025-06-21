import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from generator_model import Generator

# Load model
@st.cache_resource
def load_model():
    model = Generator()
    model.load_state_dict(torch.load("modelo_p3_contest.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

def generate_images(model, digit, num_images=5, latent_dim=100):
    noise = torch.randn(num_images, latent_dim)
    labels = torch.tensor([digit] * num_images)
    with torch.no_grad():
        generated_imgs = model(noise, labels)
    return generated_imgs

def show_images(images):
    fig, axs = plt.subplots(1, len(images), figsize=(10, 2))
    for i, img in enumerate(images):
        axs[i].imshow(img.squeeze().numpy(), cmap="gray")
        axs[i].axis("off")
    st.pyplot(fig)

# Streamlit UI
st.title("MNIST Digit Generator")
digit = st.number_input("Enter a digit (0-9):", min_value=0, max_value=9, step=1)

if st.button("Generate Images"):
    model = load_model()
    imgs = generate_images(model, digit)
    show_images(imgs)
