import streamlit as st
from ollama import Client
import time

# Initialize Ollama client
client = Client(host='http://localhost:11434')

# App configuration with custom theme
st.set_page_config(
    page_title="Nexa AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Update the CSS section with these changes
st.markdown("""
<style>
    /* Modified Sidebar styling */
    [data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 1px solid #e0e0e0;
        padding: 2rem 1rem;
    }

    /* Sidebar title */
    [data-testid="stSidebar"] h1 {
        color: #2c3e50 !important;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Sidebar text elements */
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stSubheader,
    [data-testid="stSidebar"] label {
        color: #2c3e50 !important;
    }

    /* Slider and input elements in sidebar */
    [data-testid="stSidebar"] .stSlider,
    [data-testid="stSidebar"] .stSelectbox,
    [data-testid="stSidebar"] .stTextArea {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 8px;
    }

    /* Status indicator */
    [data-testid="stSidebar"] .stAlert {
        background-color: #f8f9fa !important;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for settings
with st.sidebar:
    st.title("⚙️ Nexa Settings")

    # Divider
    st.markdown("---")

    # Model selection
    st.subheader("Model Configuration")
    available_models = ["llama3.2"]
    selected_model = st.selectbox("Select Model", available_models, index=0)

    # Configuration options
    temperature = st.slider("Creativity", 0.0, 1.0, 0.7, 0.01,
                            help="Higher values make output more random, lower values make it more deterministic")
    max_length = st.slider("Response Length", 100, 2000, 500, 50,
                           help="Maximum number of tokens to generate in the response")

    # Divider
    st.markdown("---")

    # Personality settings
    st.subheader("Personality Settings")
    system_prompt = st.text_area(
        "Customize Nexa's personality",
        "You are Nexa, a helpful AI assistant. Provide clear, concise responses. "
        "Maintain a friendly and professional tone.",
        height=150,
        help="This text defines how the AI should behave and respond"
    )

    # Divider
    st.markdown("---")

    # Status indicator
    st.subheader("System Status")
    try:
        client.generate(model=selected_model, prompt="test")
        st.success("✅ Ollama is connected and ready")
    except:
        st.error("⚠️ Ollama connection failed")

# Main chat area
st.title("💬 Nexa AI Assistant")
st.caption("Your intelligent assistant for all queries")

# Initialize chat
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hello! I'm Nexa, your AI assistant. How can I help you today?"
    })

# Chat container
with st.container():
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


# Generate response function
def generate_response(prompt):
    try:
        response = client.generate(
            model=selected_model,
            prompt=prompt,
            system=system_prompt,
            options={
                'temperature': temperature,
                'num_predict': max_length
            }
        )
        return response['response']
    except Exception as e:
        return f"⚠️ Error: {str(e)}. Please check if Ollama is running."


# Chat input
if prompt := st.chat_input("Type your message..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Get response
        assistant_response = generate_response(prompt)

        # Stream the response
        for chunk in assistant_response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})