import streamlit as st
import google.generativeai as genai
from google.oauth2 import service_account
from google.cloud import aiplatform
from vertexai.preview.vision_models import ImageGenerationModel

# Define your image generation function
def get_image(prompt, aspect_ratio):
    gcp_credentials = st.secrets["gcp_service_account"]
    credentials = service_account.Credentials.from_service_account_info(gcp_credentials)
    gcp_project_id = gcp_credentials["project_id"]
    
    # Initialize AI Platform
    aiplatform.init(project=gcp_project_id, credentials=credentials)
    model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001")
    
    # Generate image
    image = model.generate_images(prompt=prompt, aspect_ratio=aspect_ratio)
    image[0].save(location="./gen-img1.png", include_generation_parameters=True)

# Streamlit app code
st.title("Image Generator")

# Input area for description
book_description = st.text_area("Enter the Description:", height=300)
aspect_ratios = ['1:1', '9:16', '16:9', '4:3', '3:4']

# Selectbox for aspect ratio
selected_ratio = st.selectbox("Select Aspect Ratio", options=aspect_ratios, index=1)

# Button to generate the image
if st.button("Generate Image"):
    if book_description:
        with st.spinner("Generating Image..."):
            try:
                get_image(book_description, selected_ratio)
                st.image("./gen-img1.png", caption='Generated Image', use_column_width=True)
                st.success("Image generated successfully!")
            except Exception as e:
                error_message = str(e).lower()
                if "safety filter" in error_message or "prohibited words" in error_message:
                    st.error("The prompt violates the content policy. Please modify your description and try again.")
                else:
                    st.error("An error occurred while generating the image. Please try again.")
    else:
        st.error("Please enter a description to generate the image!")

