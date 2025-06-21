import streamlit as st

# Set app title
st.title("Number Input and Image Viewer")

# Number input
number = st.number_input("Enter a number", min_value=0, step=1)

# Displaying 5 images
st.subheader(f"Showing 5 images for input: {number}")

# Placeholder images (use your own URLs or file paths if needed)
image_urls = [
    "https://via.placeholder.com/150?text=Image+1",
    "https://via.placeholder.com/150?text=Image+2",
    "https://via.placeholder.com/150?text=Image+3",
    "https://via.placeholder.com/150?text=Image+4",
    "https://via.placeholder.com/150?text=Image+5",
]

for i, url in enumerate(image_urls, 1):
    st.image(url, caption=f"Image {i}", use_column_width=True)
