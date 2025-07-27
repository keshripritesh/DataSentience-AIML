## PROJECT TITLE : BIRD SPECIES CLASSIFICATION

#### AIM

To classify different species of birds using deep learning and computer vision.

#### DESCRIPTION

This is a classification problem where we classify different species of birds using a Xception Transfer Learning Model. The application provides a user-friendly Streamlit interface for real-time bird species prediction.

**✨ Recent Updates:**
- ✅ Fixed deprecated Streamlit cache functions for better performance
- ✅ Updated to modern Streamlit `@st.cache_resource` 
- ✅ Enhanced UI with improved user experience
- ✅ Added confidence visualization and top-3 predictions
- ✅ Fixed deprecated PIL Image.ANTIALIAS usage
- ✅ Added proper error handling and documentation

#### FEATURES

- 🦅 **400+ Bird Species Classification**
- 📱 **Mobile-Responsive Interface** 
- 🎯 **Real-time Predictions** with confidence scores
- 📊 **Top-3 Predictions Display**
- ⚡ **Optimized Performance** with modern caching
- 🖼️ **Image Preview** before prediction

#### LINK TO WEBAPP:

*Note: The original deployment link is no longer active. To run the application locally, please follow the setup instructions below.*

#### GLANCE AT THE WEBAPP

![bird1](https://user-images.githubusercontent.com/72400676/160309740-bf22a5e4-4887-4f08-a514-2deaf984d5e1.JPG)

![bird2](https://user-images.githubusercontent.com/72400676/160309748-b9cdc5e7-d2c1-466c-bbdd-7ad3cedeaa45.JPG)

#### MODELS USED

Xception Transfer Learning Model - 94.9 % accuracy

#### LINK TO .H5 FILE:

https://drive.google.com/file/d/1MXPAFeg029S82cZywaRyRvY9twYq0pNF/view?usp=sharing

#### DEPLOYMENT

StreamLit-Share = Streamlit turns data scripts into shareable web apps in minutes. All in Python. No front‑end experience required. Streamlit’s open-source app framework is a breeze to get started with. It is very easy to use. One just have to connect with the github repo and start deploying it on streamlit share and you are good to go.

#### LIBRARIES NEEDED

1. Numpy
2. Tensorflow
3. Keras
4. Xception
5. Glob
6. StreamLit

## 🚀 Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Clone the repository and navigate to the project directory:**
   ```bash
   cd "src/Environmental Monitoring/Bird Species Classification Web App"
   ```

2. **Create and activate a virtual environment (recommended):**
   
   **Create Python Virtual Environment:**
   ```bash
   python3 -m venv myenv
   ```
   
   **Activate the virtual environment:**
   
   On macOS and Linux:
   ```bash
   source myenv/bin/activate
   ```
   
   On Windows:
   ```bash
   myenv\Scripts\activate
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the model file:**
   
   Download the pre-trained model from: [Model Link](https://drive.google.com/file/d/1MXPAFeg029S82cZywaRyRvY9twYq0pNF/view?usp=sharing)
   
   Save it as `bird_classification_new_model.h5` in the project directory.

5. **Run the application:**
   ```bash
   streamlit run app.py
   ```

6. **Access the web app:**
   
   Open your browser and go to `http://localhost:8501`

7. **To deactivate the virtual environment when you're done:**
   ```bash
   deactivate
   ```

### Troubleshooting
- Ensure all dependencies are installed correctly
- Verify the model file is in the correct location
- Check that you're using Python 3.8 or higher

#### CONCLUSION

We can conclude that our Web App classifies bird species with the help of Xception Tranfer Learning library.

#### CONTRIBUTED BY

[Shreya Ghosh](https://github.com/shreya024)
