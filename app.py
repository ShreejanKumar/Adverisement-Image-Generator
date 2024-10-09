import streamlit as st
import vertexai
from vertexai.preview.vision_models import Image, ImageGenerationModel
from google.oauth2 import service_account
import tempfile
from google.cloud import aiplatform
from io import BytesIO


def get_image(prompt, img):
    gcp_credentials = st.secrets["gcp_service_account"]
    credentials = service_account.Credentials.from_service_account_info(gcp_credentials)
    gcp_project_id = gcp_credentials["project_id"]
    aiplatform.init(project=gcp_project_id, credentials=credentials)
    
    file_bytes = img.read()
    with open(f"./temp_{img.name}", "wb") as f:
        f.write(file_bytes)
    base_img = Image.load_from_file(location = f"./temp_{img.name}")
    model = ImageGenerationModel.from_pretrained("imagegeneration@006")
    image = model.edit_image(
                    base_image=base_img,
                    prompt=prompt,
                    edit_mode="product-image",
                    guidance_scale = 300
                )
    image[0].save(location="./output-image.png", include_generation_parameters=False)
    
# Streamlit app code
st.title("Image Generator")

# Input area for description
prompt = st.text_area("Enter the prompt", height=100)

# File uploader to take an image input
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

# Button to generate the image
if st.button("Generate Image"):
    if prompt and uploaded_file is not None:
        with st.spinner("Generating Image..."):
            try:
                # Save the generated image
                output_image_path = "./output-image.png"
                get_image(prompt,uploaded_file)
                
                # Display the generated image in Streamlit
                st.image(output_image_path, caption='Generated Image', use_column_width=True)
                st.success("Image generated successfully!")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.error("Please enter a prompt and upload an image to generate the output!")


