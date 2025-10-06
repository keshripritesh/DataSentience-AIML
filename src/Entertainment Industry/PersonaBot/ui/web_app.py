"""
Streamlit web application for PersonaBot
Interactive web interface with chat and personality visualization
"""

import streamlit as st
import sys
import os
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from typing import Dict, List, Any
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.personabot import PersonaBot
from config.settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="PersonaBot - Advanced Conversational AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .bot-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .personality-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

class PersonaBotWebApp:
    """Streamlit web application for PersonaBot"""
    
    def __init__(self):
        """Initialize the web app"""
        self.initialize_session_state()
        self.setup_sidebar()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state"""
        if 'bot' not in st.session_state:
            st.session_state.bot = None
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'personality_history' not in st.session_state:
            st.session_state.personality_history = []
        if 'performance_history' not in st.session_state:
            st.session_state.performance_history = []
    
    def setup_sidebar(self):
        """Setup the sidebar with controls"""
        st.sidebar.title("🤖 PersonaBot Controls")
        
        # Bot initialization
        if st.session_state.bot is None:
            st.sidebar.subheader("Initialize Bot")
            
            # Personality configuration
            st.sidebar.write("**Initial Personality:**")
            personality_traits = {}
            
            col1, col2 = st.sidebar.columns(2)
            with col1:
                personality_traits['humor'] = st.slider("Humor", 0.0, 1.0, 0.5, 0.1)
                personality_traits['formality'] = st.slider("Formality", 0.0, 1.0, 0.5, 0.1)
                personality_traits['empathy'] = st.slider("Empathy", 0.0, 1.0, 0.5, 0.1)
                personality_traits['sarcasm'] = st.slider("Sarcasm", 0.0, 1.0, 0.3, 0.1)
            
            with col2:
                personality_traits['enthusiasm'] = st.slider("Enthusiasm", 0.0, 1.0, 0.6, 0.1)
                personality_traits['professionalism'] = st.slider("Professionalism", 0.0, 1.0, 0.7, 0.1)
                personality_traits['creativity'] = st.slider("Creativity", 0.0, 1.0, 0.5, 0.1)
                personality_traits['assertiveness'] = st.slider("Assertiveness", 0.0, 1.0, 0.4, 0.1)
            
            enable_rl = st.sidebar.checkbox("Enable Reinforcement Learning", value=True)
            
            if st.sidebar.button("Initialize PersonaBot", type="primary"):
                with st.spinner("Initializing PersonaBot..."):
                    try:
                        st.session_state.bot = PersonaBot(
                            initial_personality=personality_traits,
                            enable_rl=enable_rl
                        )
                        st.success("PersonaBot initialized successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to initialize PersonaBot: {e}")
        
        # Bot controls
        if st.session_state.bot is not None:
            st.sidebar.subheader("Bot Controls")
            
            if st.sidebar.button("Reset Conversation"):
                st.session_state.bot.reset_conversation()
                st.session_state.chat_history = []
                st.session_state.personality_history = []
                st.session_state.performance_history = []
                st.success("Conversation reset!")
                st.rerun()
            
            if st.sidebar.button("Save Session"):
                try:
                    saved_path = st.session_state.bot.save_session()
                    st.success(f"Session saved to {saved_path}")
                except Exception as e:
                    st.error(f"Failed to save session: {e}")
            
            # Session management
            st.sidebar.subheader("Session Management")
            available_sessions = st.session_state.bot.get_available_sessions()
            
            if available_sessions:
                session_names = [os.path.basename(s) for s in available_sessions]
                selected_session = st.sidebar.selectbox("Load Session", session_names)
                
                if st.sidebar.button("Load Selected Session"):
                    try:
                        session_path = available_sessions[session_names.index(selected_session)]
                        if st.session_state.bot.load_session(session_path):
                            st.success(f"Session loaded: {selected_session}")
                            st.rerun()
                        else:
                            st.error("Failed to load session")
                    except Exception as e:
                        st.error(f"Error loading session: {e}")
            else:
                st.sidebar.write("No saved sessions found")
    
    def run(self):
        """Run the main application"""
        # Header
        st.markdown('<h1 class="main-header">🤖 PersonaBot</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Advanced Conversational AI with Dynamic Personality Adaptation</p>', unsafe_allow_html=True)
        
        # Main content
        if st.session_state.bot is None:
            self.show_welcome_page()
        else:
            self.show_main_interface()
    
    def show_welcome_page(self):
        """Show welcome page when bot is not initialized"""
        st.markdown("""
        ## Welcome to PersonaBot! 🎉
        
        PersonaBot is a cutting-edge conversational AI system that leverages **Reinforcement Learning** 
        and **Advanced NLP** to create dynamic, adaptive personalities that evolve based on user interactions.
        
        ### Key Features:
        - 🧠 **Dynamic Personality Adaptation**: Real-time personality evolution
        - 🎯 **Reinforcement Learning Engine**: Advanced RL algorithms
        - 📊 **Sentiment-Aware Responses**: Contextual understanding
        - 🔄 **Continuous Learning**: Online learning from every interaction
        - 📈 **Performance Analytics**: Real-time metrics and visualization
        
        ### Getting Started:
        1. Configure the initial personality traits in the sidebar
        2. Choose whether to enable Reinforcement Learning
        3. Click "Initialize PersonaBot" to start
        4. Begin chatting and watch the personality adapt!
        
        ---
        
        **Ready to experience the future of conversational AI?** 
        Use the sidebar to configure and initialize your PersonaBot! 🚀
        """)
    
    def show_main_interface(self):
        """Show the main chat interface"""
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📊 Personality", "📈 Performance", "🔧 Settings"])
        
        with tab1:
            self.show_chat_interface()
        
        with tab2:
            self.show_personality_interface()
        
        with tab3:
            self.show_performance_interface()
        
        with tab4:
            self.show_settings_interface()
    
    def show_chat_interface(self):
        """Show the chat interface"""
        st.subheader("💬 Chat with PersonaBot")
        
        # Chat container
        chat_container = st.container()
        
        with chat_container:
            # Display chat history
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>You:</strong> {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message bot-message">
                        <strong>🤖 PersonaBot:</strong> {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Chat input
        st.markdown("---")
        
        # Start conversation if not started
        if not st.session_state.chat_history:
            if st.button("Start Conversation", type="primary"):
                welcome_message = st.session_state.bot.start_conversation()
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": welcome_message,
                    "timestamp": datetime.now()
                })
                st.rerun()
        
        # Chat input
        if st.session_state.chat_history:
            user_input = st.chat_input("Type your message here...")
            
            if user_input:
                # Add user message
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": user_input,
                    "timestamp": datetime.now()
                })
                
                # Generate bot response
                with st.spinner("PersonaBot is thinking..."):
                    response = st.session_state.bot.chat(user_input)
                
                # Add bot response
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now()
                })
                
                # Update history
                self.update_histories()
                
                st.rerun()
    
    def show_personality_interface(self):
        """Show personality visualization interface"""
        st.subheader("📊 Personality Analysis")
        
        if not st.session_state.personality_history:
            st.info("Start a conversation to see personality analysis!")
            return
        
        # Current personality
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("**Current Personality Traits:**")
            personality_summary = st.session_state.bot.get_personality_summary()
            current_traits = personality_summary['current_traits']
            
            # Create personality radar chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=list(current_traits.values()),
                theta=list(current_traits.keys()),
                fill='toself',
                name='Current Personality',
                line_color='purple'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title="Personality Radar Chart"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**Personality Metrics:**")
            metrics = personality_summary
            st.metric("Stability", f"{metrics['stability']:.3f}")
            st.metric("Adaptation Count", metrics['adaptation_count'])
            st.metric("Drift", f"{metrics['drift']:.3f}")
            
            st.write("**Summary:**")
            st.info(metrics['summary'])
        
        # Personality evolution over time
        if len(st.session_state.personality_history) > 1:
            st.write("**Personality Evolution:**")
            
            # Create evolution chart
            df = pd.DataFrame(st.session_state.personality_history)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            fig = px.line(df, x='timestamp', y=list(current_traits.keys()),
                         title="Personality Evolution Over Time")
            fig.update_layout(xaxis_title="Time", yaxis_title="Trait Value")
            st.plotly_chart(fig, use_container_width=True)
    
    def show_performance_interface(self):
        """Show performance metrics interface"""
        st.subheader("📈 Performance Analytics")
        
        if not st.session_state.performance_history:
            st.info("Start a conversation to see performance metrics!")
            return
        
        # Current performance
        performance = st.session_state.bot.get_performance_summary()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Interactions", performance['total_interactions'])
        with col2:
            st.metric("Average Sentiment", f"{performance['average_sentiment']:.3f}")
        with col3:
            st.metric("Average Engagement", f"{performance['average_engagement']:.3f}")
        with col4:
            st.metric("Personality Adaptations", performance['personality_adaptations'])
        
        # Performance over time
        if len(st.session_state.performance_history) > 1:
            st.write("**Performance Over Time:**")
            
            df = pd.DataFrame(st.session_state.performance_history)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Create performance charts
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.line(df, x='timestamp', y='sentiment',
                             title="Sentiment Over Time")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.line(df, x='timestamp', y='engagement',
                             title="Engagement Over Time")
                st.plotly_chart(fig, use_container_width=True)
        
        # RL stats if available
        if performance['rl_stats']:
            st.write("**Reinforcement Learning Statistics:**")
            
            rl_stats = performance['rl_stats']
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Epsilon", f"{rl_stats['epsilon']:.3f}")
            with col2:
                st.metric("Memory Size", rl_stats['memory_size'])
            with col3:
                st.metric("Episodes", rl_stats['training_stats']['episodes'])
            with col4:
                st.metric("Avg Reward", f"{rl_stats['training_stats']['average_reward']:.3f}")
    
    def show_settings_interface(self):
        """Show settings and configuration interface"""
        st.subheader("🔧 Settings & Configuration")
        
        # Model information
        st.write("**Model Information:**")
        model_info = st.session_state.bot.get_model_info()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**NLP Engine:**")
            nlp_info = model_info['nlp_engine']
            for key, value in nlp_info.items():
                st.write(f"- **{key}:** {value}")
        
        with col2:
            st.write("**Personality Encoder:**")
            personality_info = model_info['personality_encoder']
            st.write(f"- **Traits:** {', '.join(personality_info['traits'])}")
            st.write(f"- **Current Values:** {personality_info['current_values']}")
        
        # Export/Import options
        st.write("**Data Management:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export Personality"):
                try:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filepath = os.path.join(settings.data.models_dir, f"personality_{timestamp}.json")
                    st.session_state.bot.export_personality(filepath)
                    st.success(f"Personality exported to {filepath}")
                except Exception as e:
                    st.error(f"Failed to export personality: {e}")
        
        with col2:
            uploaded_file = st.file_uploader("Import Personality", type=['json'])
            if uploaded_file is not None:
                try:
                    # Save uploaded file temporarily
                    temp_path = f"temp_personality_{datetime.now().timestamp()}.json"
                    with open(temp_path, 'wb') as f:
                        f.write(uploaded_file.getvalue())
                    
                    # Import personality
                    if st.session_state.bot.import_personality(temp_path):
                        st.success("Personality imported successfully!")
                        os.remove(temp_path)  # Clean up
                        st.rerun()
                    else:
                        st.error("Failed to import personality")
                        os.remove(temp_path)  # Clean up
                except Exception as e:
                    st.error(f"Error importing personality: {e}")
    
    def update_histories(self):
        """Update personality and performance histories"""
        # Update personality history
        personality_summary = st.session_state.bot.get_personality_summary()
        st.session_state.personality_history.append({
            'timestamp': datetime.now(),
            **personality_summary['current_traits']
        })
        
        # Update performance history
        performance = st.session_state.bot.get_performance_summary()
        st.session_state.performance_history.append({
            'timestamp': datetime.now(),
            'sentiment': performance['average_sentiment'],
            'engagement': performance['average_engagement'],
            'interactions': performance['total_interactions']
        })

def main():
    """Main entry point"""
    app = PersonaBotWebApp()
    app.run()

if __name__ == "__main__":
    main()
