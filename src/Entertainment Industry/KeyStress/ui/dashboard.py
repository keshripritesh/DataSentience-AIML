"""
Streamlit dashboard for KeyStress.

This module provides a web-based dashboard for:
- Real-time stress monitoring
- Session visualization
- Model training and evaluation
- Data analysis and insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime, timedelta
import time

from capture.session_recorder import SessionRecorder
from features.extractor import FeatureExtractor
from features.stress_indicators import StressIndicators
from ml.dataset import Dataset
from ml.train import ModelTrainer


class Dashboard:
    """
    Streamlit dashboard for KeyStress.
    
    This class provides a comprehensive web interface for:
    - Real-time stress monitoring
    - Session management and visualization
    - Model training and evaluation
    - Data analysis and insights
    """
    
    def __init__(self):
        """Initialize the dashboard."""
        self.session_recorder = SessionRecorder()
        self.feature_extractor = FeatureExtractor()
        self.stress_analyzer = StressIndicators()
        
        # Configure page
        st.set_page_config(
            page_title="KeyStress Dashboard",
            page_icon="🔐",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def run(self):
        """Run the dashboard."""
        # Sidebar
        self._create_sidebar()
        
        # Main content
        st.title("🔐 KeyStress Dashboard")
        st.markdown("Keystroke-based stress and fatigue detection system")
        
        # Navigation
        page = st.sidebar.selectbox(
            "Navigation",
            ["Overview", "Real-time Monitoring", "Session Analysis", "Model Training", "Data Insights"]
        )
        
        if page == "Overview":
            self._show_overview()
        elif page == "Real-time Monitoring":
            self._show_realtime_monitoring()
        elif page == "Session Analysis":
            self._show_session_analysis()
        elif page == "Model Training":
            self._show_model_training()
        elif page == "Data Insights":
            self._show_data_insights()
    
    def _create_sidebar(self):
        """Create the sidebar with controls."""
        st.sidebar.title("KeyStress Controls")
        
        # Quick actions
        st.sidebar.subheader("Quick Actions")
        
        if st.sidebar.button("Start New Session"):
            self._start_new_session()
        
        if st.sidebar.button("Refresh Data"):
            st.rerun()
        
        # Session info
        st.sidebar.subheader("Session Info")
        stats = self.session_recorder.get_session_stats()
        
        st.sidebar.metric("Total Sessions", stats['total_sessions'])
        st.sidebar.metric("Completed Sessions", stats['completed_sessions'])
        st.sidebar.metric("Total Duration", f"{stats['total_duration']:.1f}s")
    
    def _show_overview(self):
        """Show the overview page."""
        st.header("📊 Overview")
        
        # Get session statistics
        stats = self.session_recorder.get_session_stats()
        sessions = self.session_recorder.list_sessions()
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Sessions", stats['total_sessions'])
        
        with col2:
            st.metric("Completed Sessions", stats['completed_sessions'])
        
        with col3:
            st.metric("Total Duration", f"{stats['total_duration']:.1f}s")
        
        with col4:
            avg_duration = stats['avg_duration'] if stats['avg_duration'] > 0 else 0
            st.metric("Avg Duration", f"{avg_duration:.1f}s")
        
        # Recent sessions
        st.subheader("Recent Sessions")
        
        if sessions:
            # Create DataFrame for recent sessions
            recent_data = []
            for session in sessions[:10]:  # Show last 10 sessions
                recent_data.append({
                    'Session': session.session_name,
                    'Duration (s)': session.duration or 0,
                    'Stress Level': session.stress_level or 'N/A',
                    'Start Time': session.start_time.strftime('%Y-%m-%d %H:%M'),
                    'Status': 'Completed' if session.end_time else 'Active'
                })
            
            df_recent = pd.DataFrame(recent_data)
            st.dataframe(df_recent, use_container_width=True)
        else:
            st.info("No sessions recorded yet. Start a new session to begin monitoring.")
        
        # Stress level distribution
        if stats['stress_levels']:
            st.subheader("Stress Level Distribution")
            
            stress_data = []
            for level, count in stats['stress_levels'].items():
                stress_data.append({
                    'Stress Level': f'Level {level}',
                    'Count': count
                })
            
            df_stress = pd.DataFrame(stress_data)
            
            fig = px.pie(df_stress, values='Count', names='Stress Level', 
                        title='Distribution of Stress Levels')
            st.plotly_chart(fig, use_container_width=True)
    
    def _show_realtime_monitoring(self):
        """Show real-time monitoring page."""
        st.header("📈 Real-time Monitoring")
        
        # Monitoring controls
        col1, col2 = st.columns(2)
        
        with col1:
            session_name = st.text_input("Session Name", value=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        with col2:
            duration = st.number_input("Duration (seconds)", min_value=30, value=300, step=30)
        
        # Start monitoring
        if st.button("Start Monitoring", type="primary"):
            st.info("Real-time monitoring functionality requires integration with keylogger")
            st.info("This would start keystroke capture and display real-time stress indicators")
        
        # Simulated real-time data
        st.subheader("Simulated Real-time Stress Level")
        
        # Create simulated data
        time_points = pd.date_range(start=datetime.now() - timedelta(minutes=10), 
                                  end=datetime.now(), freq='30S')
        
        # Simulate stress levels
        np.random.seed(42)
        stress_levels = np.random.normal(0.5, 0.2, len(time_points))
        stress_levels = np.clip(stress_levels, 0, 1)
        
        # Create real-time chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time_points,
            y=stress_levels,
            mode='lines+markers',
            name='Stress Level',
            line=dict(color='red', width=2),
            fill='tonexty'
        ))
        
        fig.update_layout(
            title='Real-time Stress Level',
            xaxis_title='Time',
            yaxis_title='Stress Level (0-1)',
            yaxis=dict(range=[0, 1]),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Current stress indicator
        current_stress = stress_levels[-1]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if current_stress < 0.33:
                st.success(f"Low Stress: {current_stress:.2f}")
            elif current_stress < 0.67:
                st.warning(f"Medium Stress: {current_stress:.2f}")
            else:
                st.error(f"High Stress: {current_stress:.2f}")
        
        with col2:
            st.metric("Typing Speed", "180 CPM")
        
        with col3:
            st.metric("Error Rate", "5.2%")
    
    def _show_session_analysis(self):
        """Show session analysis page."""
        st.header("🔍 Session Analysis")
        
        # Session selection
        sessions = self.session_recorder.list_sessions()
        
        if not sessions:
            st.info("No sessions available for analysis.")
            return
        
        session_names = [s.session_name for s in sessions]
        selected_session = st.selectbox("Select Session", session_names)
        
        if selected_session:
            self._analyze_session(selected_session)
    
    def _analyze_session(self, session_name: str):
        """Analyze a specific session."""
        # Load session data
        log_file = f"data/logs/{session_name}.json"
        
        if not os.path.exists(log_file):
            st.error(f"Log file not found for session: {session_name}")
            return
        
        try:
            # Extract features
            features = self.feature_extractor.extract_features_from_file(log_file)
            
            # Generate stress report
            report = self.stress_analyzer.generate_stress_report(features, session_name)
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Stress Analysis")
                st.metric("Stress Score", f"{report['stress_score']:.3f}")
                st.metric("Stress Level", report['stress_level'])
                st.metric("Stressful Indicators", report['stressful_indicators_count'])
            
            with col2:
                st.subheader("Top Contributors")
                for contrib in report['top_contributors']:
                    st.write(f"• {contrib['category']}: {contrib['contribution']:.3f}")
            
            # Feature breakdown
            st.subheader("Feature Breakdown")
            
            # Create feature comparison chart
            feature_categories = {
                'Speed': ['avg_interval', 'cpm', 'wpm'],
                'Errors': ['error_rate', 'backspace_rate', 'corrections_per_100'],
                'Pauses': ['pause_rate', 'avg_pause_duration', 'long_pause_rate'],
                'Variability': ['interval_std', 'interval_cv', 'interval_range']
            }
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=list(feature_categories.keys()),
                specs=[[{"type": "bar"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "bar"}]]
            )
            
            for i, (category, feature_list) in enumerate(feature_categories.items()):
                row = (i // 2) + 1
                col = (i % 2) + 1
                
                values = [features.get(f, 0) for f in feature_list]
                
                fig.add_trace(
                    go.Bar(x=feature_list, y=values, name=category),
                    row=row, col=col
                )
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.subheader("Recommendations")
            for rec in report['recommendations']:
                st.info(rec)
                
        except Exception as e:
            st.error(f"Error analyzing session: {e}")
    
    def _show_model_training(self):
        """Show model training page."""
        st.header("🤖 Model Training")
        
        # Training controls
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.selectbox(
                "Model Type",
                ["random_forest", "logistic", "svm"],
                format_func=lambda x: x.replace('_', ' ').title()
            )
        
        with col2:
            hyperparameter_tuning = st.checkbox("Enable Hyperparameter Tuning")
        
        # Training button
        if st.button("Train Model", type="primary"):
            with st.spinner("Training model..."):
                try:
                    trainer = ModelTrainer()
                    
                    # Prepare data
                    if not trainer.prepare_data():
                        st.error("Failed to prepare data for training")
                        return
                    
                    # Train model
                    results = trainer.train_model(
                        model_type=model_type,
                        hyperparameter_tuning=hyperparameter_tuning
                    )
                    
                    st.success("Model training completed!")
                    
                    # Display results
                    accuracy = results['evaluation_results']['accuracy']
                    st.metric("Model Accuracy", f"{accuracy:.3f}")
                    
                    # Show detailed results
                    with st.expander("Detailed Results"):
                        st.json(results['evaluation_results'])
                        
                except Exception as e:
                    st.error(f"Error training model: {e}")
        
        # Model comparison
        st.subheader("Model Comparison")
        
        # Simulated model comparison data
        models = ['Random Forest', 'Logistic Regression', 'SVM']
        accuracies = [0.85, 0.78, 0.82]
        
        fig = px.bar(
            x=models,
            y=accuracies,
            title="Model Performance Comparison",
            labels={'x': 'Model', 'y': 'Accuracy'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _show_data_insights(self):
        """Show data insights page."""
        st.header("📊 Data Insights")
        
        # Load dataset
        dataset = Dataset()
        
        try:
            if dataset.load_data(min_sessions=1):
                # Feature importance
                st.subheader("Feature Importance")
                
                feature_stats = dataset.get_feature_importance_data()
                
                if not feature_stats.empty:
                    # Get correlation with stress level
                    correlations = feature_stats.loc['correlation']
                    
                    # Create feature importance chart
                    fig = px.bar(
                        x=correlations.values,
                        y=correlations.index,
                        orientation='h',
                        title="Feature Correlation with Stress Level"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Dataset statistics
                st.subheader("Dataset Statistics")
                
                summary = dataset.get_session_summary()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Total Sessions", summary['total_sessions'])
                    st.metric("Features", summary['feature_count'])
                
                with col2:
                    st.metric("Completed Sessions", summary['completed_sessions'])
                    
                    if summary['label_distribution']:
                        st.write("Label Distribution:")
                        for label, count in summary['label_distribution'].items():
                            st.write(f"  Level {label}: {count}")
            
            else:
                st.info("No data available for insights. Record some sessions first.")
                
        except Exception as e:
            st.error(f"Error loading data: {e}")
    
    def _start_new_session(self):
        """Start a new monitoring session."""
        st.info("New session functionality would integrate with the keylogger")
        st.info("This would open a new window/tab for keystroke monitoring")


def main():
    """Main function to run the dashboard."""
    dashboard = Dashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
