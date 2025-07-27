import streamlit as st
from yolo_code import load_model, train_model  # Import functions from yolo_code.py


# Load the trained YOLO model
model_path = "yolo11/runs/detect/train2/weights/best.pt"  # Update this path according to your system
model = yolo(model_path)

# Streamlit UI
st.title("Brain Tumor Detection using YOLO")
st.write("Upload an MRI image ")
# Image uploader
uploaded_image = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Save the uploaded image temporarily
    temp_image_path = "temp_image.png"
    image.save(temp_image_path)

    # Run the model prediction
    st.write("Detecting tumor...")
    
    try:
        results = model(temp_image_path)  # Perform inference
        
        # Create output directory if it doesn't exist
        output_dir = "outputs"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save and display the detection result
        results.save(save_dir=output_dir)  # Save result in 'outputs' folder
        output_image_path = os.path.join(output_dir, os.path.basename(temp_image_path))

        # Check if the output image was saved successfully
        if os.path.exists(output_image_path):
            # Display the detection result
            st.image(output_image_path, caption="Detection Results", use_column_width=True)
            st.success("Tumor detection completed!")
        else:
            st.error("Error: Detection result not found.")
    
    except Exception as e:
        st.error(f"An error occurred during detection: {e}")
