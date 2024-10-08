import streamlit as st
import vertexai
from vertexai.preview.vision_models import Image, ImageGenerationModel
from google.oauth2 import service_account
import tempfile
from io import BytesIO

# Initialize GCP credentials and Vertex AI
gcp_credentials = st.secrets["gcp_service_account"]
credentials = service_account.Credentials.from_service_account_info(gcp_credentials)
PROJECT_ID = gcp_credentials["project_id"]
vertexai.init(project=PROJECT_ID, location="us-central1")

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
                # Save the uploaded image to a temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    input_image_path = tmp_file.name

                # Load the image using vertexai
                base_img = Image.load_from_file(location=input_image_path)

                # Load the image generation model
                model = ImageGenerationModel.from_pretrained("imagegeneration@006")

                # Generate the edited image based on the prompt
                images = model.edit_image(
                    base_image=base_img,
                    prompt=prompt,
                    edit_mode="product-image",
                )

                # Save the generated image
                output_image_path = "./output-image.png"
                images[0].save(location=output_image_path, include_generation_parameters=False)

                # Display the generated image in Streamlit
                st.image(output_image_path, caption='Generated Image', use_column_width=True)
                st.success("Image generated successfully!")

                print(f"Created output image using {len(images[0]._image_bytes)} bytes")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.error("Please enter a prompt and upload an image to generate the output!")
