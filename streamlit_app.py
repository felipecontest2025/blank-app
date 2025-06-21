import streamlit as st
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import random

# Load MNIST dataset
@st.cache_data
def load_mnist():
    transform = transforms.ToTensor()
    mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    return mnist_data

mnist_data = load_mnist()

st.title("MNIST Digit Viewer")
digit = st.number_input("Enter a digit (0-9):", min_value=0, max_value=9, step=1)

# Filter images matching the selected digit
matching_images = [img for img, label in mnist_data if label == digit]
selected_images = random.sample(matching_images, k=5)

# Display images
st.subheader(f"5 examples of digit: {digit}")
cols = st.columns(5)

for idx, col in enumerate(cols):
    img = selected_images[idx].squeeze(0).numpy()
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    ax.axis('off')
    col.pyplot(fig)
