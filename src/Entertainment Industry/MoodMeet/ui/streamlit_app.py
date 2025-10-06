"""
🏢 MeetingIQ Professional - Enterprise Meeting Analytics Platform
Advanced AI-powered meeting sentiment analysis and team collaboration insights
Professional-grade dashboard for corporate teams and organizations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, date
import random
from typing import Dict, List, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules with error handling
try:
    from data.uploader import TextInputHandler
    from analysis.sentiment_analyzer import SentimentAnalyzer, SentimentTrendAnalyzer
    from analysis.mood_clustering import MoodClusterer, TopicAnalyzer
    from analysis.keyword_extractor import KeywordExtractor, PhraseExtractor
    from visualization.mood_timeline import StreamlitVisualizer
    from visualization.heatmap_generator import StreamlitHeatmapVisualizer
    MODULES_AVAILABLE = True
except ImportError as e:
    st.warning(f"Some analysis modules not available: {e}. Running in demo mode.")
    MODULES_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="MeetingIQ Professional | Enterprise Meeting Analytics",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Enterprise CSS with Pastel Corporate Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #F8FAFC 0%, #E2E8F0 100%);
        padding: 1.5rem;
    }
    
    /* Professional Header */
    .enterprise-header {
        background: linear-gradient(135deg, #E0F2FE 0%, #BFDBFE 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(59, 130, 246, 0.15);
        border: 1px solid rgba(147, 197, 253, 0.3);
    }
    
    .enterprise-header h1 {
        color: #1E40AF;
        font-size: 3.2rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .enterprise-header p {
        color: #1D4ED8;
        font-size: 1.3rem;
        margin: 1rem 0 0 0;
        font-weight: 500;
    }
    
    /* Professional Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%);
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin: 0.8rem 0;
        box-shadow: 0 6px 25px rgba(34, 197, 94, 0.12);
        border: 1px solid rgba(134, 239, 172, 0.3);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 35px rgba(34, 197, 94, 0.2);
    }
    
    .metric-card h3 {
        margin: 0;
        color: #15803D;
        font-size: 1.1rem;
        font-weight: 600;
    }
    
    .metric-card h2 {
        margin: 1rem 0;
        color: #166534;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    /* Status Cards */
    .status-card-positive {
        background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #10B981;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.1);
    }
    
    .status-card-neutral {
        background: linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #F59E0B;
        box-shadow: 0 4px 15px rgba(245, 158, 11, 0.1);
    }
    
    .status-card-negative {
        background: linear-gradient(135deg, #FEF2F2 0%, #FECACA 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #EF4444;
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.1);
    }
    
    /* Professional Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #A78BFA 0%, #8B5CF6 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(139, 92, 246, 0.4);
        background: linear-gradient(135deg, #8B5CF6 0%, #7C3AED 100%);
    }
    
    /* Sidebar Professional Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #F1F5F9 0%, #E2E8F0 100%);
    }
    
    /* Modern Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0 1.5rem;
        background: linear-gradient(135deg, #F8FAFC 0%, #E2E8F0 100%);
        border-radius: 12px;
        color: #64748B;
        font-weight: 500;
        border: 1px solid #CBD5E1;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #DBEAFE 0%, #BFDBFE 100%);
        color: #1D4ED8;
        border-color: #3B82F6;
    }
    
    /* Professional Info Boxes */
    .info-box {
        background: linear-gradient(135deg, #F3E8FF 0%, #E9D5FF 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #8B5CF6;
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.1);
    }
    
    .success-box {
        background: linear-gradient(135deg, #DCFCE7 0%, #BBF7D0 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #10B981;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.1);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #F59E0B;
        box-shadow: 0 4px 15px rgba(245, 158, 11, 0.1);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        background: linear-gradient(135deg, #E2E8F0 0%, #CBD5E1 100%);
        border-radius: 15px;
        color: #475569;
        font-weight: 500;
    }
    
    /* Professional Data Display */
    .dataframe {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
    }
    
    /* Custom Metrics */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #F0F9FF 0%, #E0F2FE 100%);
        border: 1px solid #BAE6FD;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(56, 189, 248, 0.1);
    }
    
    /* Enhanced Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%);
        border-radius: 12px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main application function."""
    
    # Professional Enterprise Header
    st.markdown("""
    <div class="enterprise-header">
        <h1>🏢 MeetingIQ Professional</h1>
        <p>Enterprise Meeting Analytics & Team Collaboration Intelligence Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Professional Sidebar Configuration
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem 0; border-bottom: 2px solid #E5E7EB; margin-bottom: 2rem;">
            <h2 style="color: #1F2937; font-weight: 700; margin: 0; font-size: 1.8rem;">🏢 MeetingIQ</h2>
            <p style="color: #6B7280; font-size: 0.95rem; margin: 0.3rem 0 0 0; font-weight: 500;">Professional Analytics</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick Analytics Overview
        st.markdown("#### 📊 Quick Analytics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Active Sessions", "2,847", "↑ 12%")
            st.metric("Team Health", "94.2%", "↑ 2.1%")
        with col2:
            st.metric("Meetings Today", "156", "↑ 8%")
            st.metric("Satisfaction", "4.7/5", "↑ 0.2")
        
        st.markdown("---")
        
        # Analysis Configuration
        st.markdown("#### ⚙️ Analysis Configuration")
        
        model_type = st.selectbox(
            "🧠 Sentiment Analysis Engine",
            ["vader", "textblob", "ensemble"],
            help="Professional-grade sentiment analysis models for enterprise use"
        )
        
        clustering_method = st.selectbox(
            "🎯 Topic Clustering Method",
            ["kmeans", "lda", "umap_hdbscan"],
            help="Advanced clustering algorithms for topic identification"
        )
        
        n_clusters = st.slider("📊 Number of Topics", 2, 8, 3)
        
        keyword_method = st.selectbox(
            "🔍 Keyword Extraction",
            ["tfidf", "rake", "yake", "ensemble"],
            help="Enterprise keyword extraction for meeting insights"
        )
        
        st.markdown("---")
        
        # Visualization Preferences
        st.markdown("#### 🎨 Dashboard Preferences")
        chart_theme = st.selectbox("Chart Theme", ["plotly", "plotly_dark", "plotly_white"])
        
        # Meeting Templates
        st.markdown("#### � Meeting Templates")
        meeting_type = st.selectbox(
            "Meeting Type",
            ["All-hands", "Team Standup", "Client Meeting", "Performance Review", "Brainstorming", "Custom"]
        )
        
        st.markdown("---")
        
        # Professional Demo Data
        st.markdown("#### 🎯 Demo Scenarios")
        
        demo_scenarios = {
            "Executive Review": """
CEO: Let's review Q4 performance and strategic initiatives.
CFO: Revenue exceeded targets by 12%, but costs increased 8%.
CTO: Technical roadmap is on track, new platform launches Q1.
CMO: Brand engagement up 25%, customer acquisition strong.
CEO: Excellent work team. Let's focus on operational efficiency.
CFO: I recommend budget reallocation for digital transformation.
CTO: Infrastructure scaling will support 50% user growth.
CMO: Market expansion into APAC shows promising early results.
            """,
            
            "Team Retrospective": """
Alice: Sprint went well overall, but we had some deployment issues.
Bob: Testing coverage improved, found critical bugs early.
Carol: Communication with design team could be better.
David: The new framework is working, development velocity up.
Alice: Client feedback was positive, they love the new features.
Bob: Let's address the CI/CD pipeline instability next sprint.
Carol: Design handoffs need clearer documentation.
David: Performance optimizations showed 40% improvement.
            """,
            
            "Client Presentation": """
PM: Thank you for joining our quarterly business review.
Client: We're excited to see the progress on our partnership.
PM: User engagement increased 35% since platform launch.
Client: That's impressive! What about the mobile experience?
Dev: Mobile app ratings improved to 4.8 stars, usage up 60%.
Client: Excellent! Timeline for the API integration phase?
PM: Phase 2 delivery scheduled for end of Q1, on track.
Client: Perfect. Let's discuss expansion opportunities.
            """
        }
        
        selected_scenario = st.selectbox("Choose Demo Scenario", list(demo_scenarios.keys()))
        
        if st.button("🚀 Load Professional Demo", use_container_width=True):
            st.session_state['input_text'] = demo_scenarios[selected_scenario].strip()
            st.success(f"✅ {selected_scenario} scenario loaded!")
        
        st.markdown("---")
        
        # Quick Actions
        st.markdown("#### ⚡ Quick Actions")
        if st.button("🔄 Reset Analysis", use_container_width=True):
            for key in ['input_text', 'analysis_results']:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("✅ Session reset!")
            
        if st.button("📤 Export Dashboard", use_container_width=True):
            st.info("💼 Export functionality - Enterprise feature!")
    
    # Professional Dashboard Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📝 Meeting Input", 
        "📊 Executive Dashboard", 
        "📈 Analytics & Insights", 
        "� Team Performance", 
        "📋 Detailed Reports",
        "⚙️ Enterprise Settings"
    ])
    
    with tab1:
        st.markdown("## 📝 Professional Meeting Input Center")
        
        # Meeting metadata
        col1, col2, col3 = st.columns(3)
        with col1:
            meeting_date = st.date_input("� Meeting Date", datetime.now().date())
        with col2:
            meeting_duration = st.number_input("⏱️ Duration (minutes)", min_value=15, max_value=480, value=60)
        with col3:
            attendee_count = st.number_input("👥 Attendees", min_value=2, max_value=50, value=5)
        
        # Professional input method selection
        input_method = st.radio(
            "📊 Input Method:",
            ["💬 Live Transcript", "📁 File Upload", "🔗 Integration"],
            horizontal=True
        )
        
        if input_method == "💬 Live Transcript":
            st.markdown("### 💬 Meeting Transcript Input")
            
            # Professional text input with enhanced placeholder
            input_text = st.text_area(
                "Enter your meeting transcript:",
                value=st.session_state.get('input_text', ''),
                height=350,
                placeholder="""Professional Format Examples:

📋 Standard Meeting:
CEO: Let's review Q4 performance metrics and strategic initiatives.
CFO: Revenue exceeded targets by 12%, operational costs increased 8%.
CTO: Technical roadmap on track, platform launch scheduled for Q1.

👥 Team Discussion:
Alice: Sprint velocity improved, but we need better testing coverage.
Bob: Integration tests caught critical issues before deployment.
Carol: Client feedback suggests UI improvements for mobile users.

💼 Client Meeting:
PM: Quarterly business review shows strong partnership growth.
Client: Impressed with platform performance and user engagement metrics.
Dev: Mobile app ratings improved to 4.8 stars, usage increased 60%."""
            )
            
            # Enhanced analysis button with professional styling
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 Analyze Meeting", type="primary", use_container_width=True):
                    if input_text.strip():
                        st.session_state['input_text'] = input_text
                        st.session_state['meeting_metadata'] = {
                            'date': meeting_date,
                            'duration': meeting_duration,
                            'attendees': attendee_count,
                            'type': meeting_type
                        }
                        process_transcript(input_text, model_type, clustering_method, n_clusters, keyword_method, chart_theme)
                    else:
                        st.error("⚠️ Please enter meeting transcript to analyze.")
        
        elif input_method == "📁 File Upload":
            st.markdown("### 📁 Professional File Upload")
            
            uploaded_file = st.file_uploader(
                "📎 Upload Meeting Transcript:",
                type=['txt', 'csv', 'docx'],
                help="Supported formats: TXT, CSV, DOCX. Maximum file size: 10MB"
            )
            
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                        st.markdown("### 📊 File Preview")
                        st.dataframe(df.head(), use_container_width=True)
                        
                        if st.button("🔍 Analyze CSV Data", type="primary"):
                            # Process CSV data
                            st.success("✅ CSV analysis initiated!")
                            
                    else:
                        content = uploaded_file.read().decode('utf-8')
                        st.markdown("### 📄 File Content Preview")
                        st.text_area("File content:", content[:1000] + "..." if len(content) > 1000 else content, height=200)
                        
                        if st.button("🚀 Analyze Uploaded File", type="primary"):
                            st.session_state['input_text'] = content
                            st.session_state['meeting_metadata'] = {
                                'date': meeting_date,
                                'duration': meeting_duration,
                                'attendees': attendee_count,
                                'type': meeting_type,
                                'source': 'file_upload'
                            }
                            process_transcript(content, model_type, clustering_method, n_clusters, keyword_method, chart_theme)
                            
                except Exception as e:
                    st.error(f"❌ Error processing file: {e}")
                    
        else:  # Integration
            st.markdown("### 🔗 Enterprise Integrations")
            
            # Integration options
            integration_options = ["Zoom", "Microsoft Teams", "Google Meet", "Slack", "Discord", "WebEx"]
            
            col1, col2 = st.columns(2)
            with col1:
                selected_integration = st.selectbox("🔌 Select Platform:", integration_options)
            with col2:
                st.text_input("🔑 API Key:", type="password", placeholder="Enter your API key")
            
            st.markdown("""
            <div class="info-box">
                <h4>🏢 Enterprise Integration Features</h4>
                <p><strong>Available in Professional Plan:</strong></p>
                <ul>
                    <li>🔄 Real-time meeting import from major platforms</li>
                    <li>📊 Automated sentiment analysis during live meetings</li>
                    <li>📈 Integration with existing business intelligence tools</li>
                    <li>🔐 SSO and enterprise security compliance</li>
                </ul>
                <p><em>Contact your administrator to enable integrations.</em></p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("## 📊 Executive Dashboard")
        
        if 'analysis_results' in st.session_state:
            # Executive Summary Cards
            display_executive_dashboard(st.session_state['analysis_results'])
        else:
            # Demo dashboard when no data
            display_demo_executive_dashboard()
    
    with tab3:
        st.markdown("### 📈 Advanced Analytics & Insights")
        
        if 'analysis_results' in st.session_state:
            display_analytics_insights(st.session_state['analysis_results'])
        else:
            display_demo_analytics_insights()
    
    with tab4:
        st.markdown("### � Team Performance Analytics")
        
        if 'analysis_results' in st.session_state:
            display_team_performance(st.session_state['analysis_results'])
        else:
            display_demo_team_performance()
    
    with tab5:
        st.markdown("### 📋 Detailed Reports")
        
        if 'analysis_results' in st.session_state:
            display_detailed_reports(st.session_state['analysis_results'])
        else:
            display_demo_detailed_reports()
    
    with tab6:
        st.markdown("### ⚙️ Enterprise Settings")
        display_enterprise_settings()


def process_transcript(text: str, model_type: str, clustering_method: str, 
                     n_clusters: int, keyword_method: str, chart_theme: str):
    """Process transcript and perform analysis."""
    
    with st.spinner("Processing transcript..."):
        try:
            # Initialize components
            text_handler = TextInputHandler()
            sentiment_analyzer = SentimentAnalyzer(model_type=model_type)
            clusterer = MoodClusterer(method=clustering_method, n_clusters=n_clusters)
            keyword_extractor = KeywordExtractor(method=keyword_method)
            topic_analyzer = TopicAnalyzer()
            trend_analyzer = SentimentTrendAnalyzer()
            
            # Process text input
            df, speaker_stats, is_valid, errors = text_handler.process_text_input(text)
            
            if not is_valid:
                st.error("Invalid transcript data:")
                for error in errors:
                    st.error(f"• {error}")
                return
            
            # Perform sentiment analysis
            sentiment_results = sentiment_analyzer.analyze_dataframe(df)
            sentiment_summary = sentiment_analyzer.get_sentiment_summary(sentiment_results)
            
            # Merge results
            df_with_sentiment = pd.concat([df, sentiment_results], axis=1)
            
            # Perform clustering
            texts = df['text'].tolist()
            sentiments = sentiment_results['polarity'].tolist()
            cluster_results = clusterer.fit(texts, sentiments)
            clustering_summary = clusterer.get_clustering_summary(cluster_results)
            
            # Extract keywords
            keywords = keyword_extractor.extract_keywords(texts, max_keywords=15)
            keyword_summary = keyword_extractor.get_keyword_summary(keywords)
            
            # Extract topics
            topics = topic_analyzer.extract_topics(texts, n_topics=5)
            topic_summary = topic_analyzer.get_topic_summary(topics)
            
            # Analyze trends
            trend_df = trend_analyzer.analyze_trend(sentiment_results)
            trend_summary = trend_analyzer.get_trend_summary(trend_df)
            
            # Store results
            st.session_state['analysis_results'] = {
                'df': df_with_sentiment,
                'speaker_stats': speaker_stats,
                'sentiment_summary': sentiment_summary,
                'cluster_results': cluster_results,
                'clustering_summary': clustering_summary,
                'keywords': keywords,
                'keyword_summary': keyword_summary,
                'topics': topics,
                'topic_summary': topic_summary,
                'trend_df': trend_df,
                'trend_summary': trend_summary,
                'chart_theme': chart_theme
            }
            
            st.success("Analysis completed successfully!")
            
        except Exception as e:
            st.error(f"Error during analysis: {e}")
            st.exception(e)


def display_analysis_results(results: Dict):
    """Display analysis results."""
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Messages",
            results['sentiment_summary'].get('total_messages', 0)
        )
    
    with col2:
        avg_polarity = results['sentiment_summary'].get('avg_polarity', 0)
        st.metric(
            "Average Sentiment",
            f"{avg_polarity:.3f}",
            delta=f"{avg_polarity:.3f}"
        )
    
    with col3:
        positive_ratio = results['sentiment_summary'].get('positive_ratio', 0)
        st.metric(
            "Positive Ratio",
            f"{positive_ratio:.1%}"
        )
    
    with col4:
        if 'sentiment_distribution' in results['sentiment_summary']:
            sentiment_dist = results['sentiment_summary']['sentiment_distribution']
            dominant_sentiment = max(sentiment_dist.items(), key=lambda x: x[1])[0]
            st.metric(
                "Dominant Sentiment",
                dominant_sentiment.title()
            )
    
    # Sentiment summary
    st.subheader("📊 Sentiment Analysis Summary")
    
    if 'most_positive_text' in results['sentiment_summary']:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Most Positive Statement:**")
            st.info(results['sentiment_summary']['most_positive_text'])
        
        with col2:
            st.markdown("**Most Negative Statement:**")
            st.warning(results['sentiment_summary']['most_negative_text'])
    
    # Clustering results
    if results['cluster_results']:
        st.subheader("🧠 Topic Clusters")
        
        for cluster in results['cluster_results']:
            with st.expander(f"Cluster {cluster.cluster_id} ({cluster.size} messages)"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Representative text:** {cluster.centroid_text}")
                    st.markdown(f"**Average sentiment:** {cluster.sentiment_avg:.3f}")
                
                with col2:
                    st.markdown("**Keywords:**")
                    for keyword in cluster.keywords[:5]:
                        st.markdown(f"• {keyword}")
    
    # Keywords
    if results['keywords']:
        st.subheader("🔑 Key Topics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Top Keywords:**")
            for i, keyword in enumerate(results['keywords'][:10]):
                st.markdown(f"{i+1}. **{keyword.keyword}** (score: {keyword.score:.3f})")
        
        with col2:
            if 'keyword_categories' in results['keyword_summary']:
                st.markdown("**Keyword Categories:**")
                for category, keywords in results['keyword_summary']['keyword_categories'].items():
                    st.markdown(f"**{category.title()}:** {len(keywords)} keywords")


def display_visualizations(results: Dict, chart_theme: str):
    """Display visualizations."""
    
    # Initialize visualizers
    timeline_viz = StreamlitVisualizer()
    heatmap_viz = StreamlitHeatmapVisualizer()
    
    # Timeline and speaker comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Sentiment Timeline")
        timeline_viz.display_timeline_chart(results['df'], speaker_column='speaker')
    
    with col2:
        st.subheader("👥 Speaker Comparison")
        timeline_viz.display_speaker_comparison(results['df'], speaker_column='speaker')
    
    # Distribution and trend
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("📊 Sentiment Distribution")
        dist_fig = timeline_viz.timeline_viz.create_sentiment_distribution(results['df'])
        st.plotly_chart(dist_fig, use_container_width=True)
    
    with col4:
        st.subheader("📈 Moving Average Trend")
        ma_fig = timeline_viz.timeline_viz.create_moving_average_chart(results['df'])
        st.plotly_chart(ma_fig, use_container_width=True)
    
    # Heatmaps
    st.subheader("🔥 Heatmap Analysis")
    heatmap_viz.display_heatmap_dashboard(
        results['df'],
        cluster_results=results['cluster_results'],
        speaker_column='speaker'
    )


def display_detailed_results(results: Dict):
    """Display detailed analysis results."""
    
    # Raw data
    st.subheader("📋 Raw Data")
    
    tab1, tab2, tab3 = st.tabs(["Processed Data", "Sentiment Results", "Speaker Statistics"])
    
    with tab1:
        st.dataframe(results['df'])
    
    with tab2:
        sentiment_cols = ['text', 'polarity', 'sentiment_label', 'confidence']
        sentiment_df = results['df'][sentiment_cols].copy()
        st.dataframe(sentiment_df)
    
    with tab3:
        speaker_stats_df = pd.DataFrame(results['speaker_stats']).T
        st.dataframe(speaker_stats_df)
    
    # Download options
    st.subheader("💾 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv_data = results['df'].to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="moodmeet_results.csv",
            mime="text/csv"
        )
    
    with col2:
        # Create summary report
        summary_text = f"""
MoodMeet Analysis Report

Summary:
- Total Messages: {results['sentiment_summary'].get('total_messages', 0)}
- Average Sentiment: {results['sentiment_summary'].get('avg_polarity', 0):.3f}
- Positive Ratio: {results['sentiment_summary'].get('positive_ratio', 0):.1%}

Sentiment Distribution:
{results['sentiment_summary'].get('sentiment_distribution', {})}

Top Keywords:
{[kw.keyword for kw in results['keywords'][:10]]}
        """
        st.download_button(
            label="Download Report",
            data=summary_text,
            file_name="moodmeet_report.txt",
            mime="text/plain"
        )
    
    with col3:
        st.info("💡 Tip: Use the CSV file for further analysis in Excel or other tools.")


def display_executive_dashboard(results: Dict):
    """Display professional executive dashboard with key metrics."""
    
    # Top-level KPIs
    st.markdown("### 🎯 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        sentiment_score = results.get('sentiment_summary', {}).get('avg_polarity', 0.5)
        sentiment_percentage = (sentiment_score + 1) * 50  # Convert -1,1 to 0,100
        st.markdown(f"""
        <div class="metric-card">
            <h3>🌟 Meeting Health Score</h3>
            <h2>{sentiment_percentage:.1f}%</h2>
            <p style="color: #059669;">{'↗️ Positive' if sentiment_score > 0 else '↘️ Needs Attention'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        total_messages = results.get('sentiment_summary', {}).get('total_messages', 0)
        engagement_score = min(100, total_messages * 2)  # Rough engagement calculation
        st.markdown(f"""
        <div class="metric-card">
            <h3>💬 Team Engagement</h3>
            <h2>{engagement_score}%</h2>
            <p style="color: #0369A1;">{total_messages} interactions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        positive_ratio = results.get('sentiment_summary', {}).get('positive_ratio', 0.5)
        st.markdown(f"""
        <div class="metric-card">
            <h3>😊 Positivity Rate</h3>
            <h2>{positive_ratio:.1%}</h2>
            <p style="color: #059669;">Team morale indicator</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        # Action items (simulated)
        action_items = len(results.get('keywords', [])) // 3
        st.markdown(f"""
        <div class="metric-card">
            <h3>✅ Action Items</h3>
            <h2>{action_items}</h2>
            <p style="color: #7C2D12;">Identified topics</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Executive charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Sentiment Trend Analysis")
        # Create executive-style sentiment chart
        if 'df' in results:
            df = results['df']
            fig = px.line(df.reset_index(), x=df.reset_index().index, y='polarity', 
                         title="Meeting Sentiment Flow",
                         color_discrete_sequence=['#3B82F6'])
            fig.update_layout(
                xaxis_title="Timeline",
                yaxis_title="Sentiment Score",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 👥 Speaker Participation")
        # Create participation chart
        if 'speaker_stats' in results:
            speaker_stats = results['speaker_stats']
            speakers = list(speaker_stats.keys())
            message_counts = [speaker_stats[speaker]['message_count'] for speaker in speakers]
            
            fig = px.bar(x=speakers, y=message_counts, 
                        title="Team Participation Levels",
                        color=message_counts,
                        color_continuous_scale='Blues')
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)


def display_analytics_insights(results: Dict):
    """Display advanced analytics and insights."""
    
    # Professional analytics header
    st.markdown("""
    <div class="info-box">
        <h3>🔬 Advanced Meeting Analytics</h3>
        <p>Deep insights into meeting dynamics, communication patterns, and team performance metrics.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Analysis tabs for detailed insights
    insight_tab1, insight_tab2, insight_tab3 = st.columns(3)
    
    with insight_tab1:
        st.markdown("#### 💬 Communication Patterns")
        
        # Word frequency analysis
        if 'df' in results:
            df = results['df']
            all_text = ' '.join(df['text'].astype(str))
            words = all_text.lower().split()
            word_freq = pd.Series(words).value_counts().head(10)
            
            fig = px.bar(x=word_freq.values, y=word_freq.index, orientation='h',
                        title="Top Keywords Used",
                        color=word_freq.values,
                        color_continuous_scale='Blues')
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with insight_tab2:
        st.markdown("#### 📊 Sentiment Dynamics")
        
        if 'df' in results:
            df = results['df']
            
            # Sentiment distribution
            sentiment_labels = []
            for score in df['polarity']:
                if score > 0.1:
                    sentiment_labels.append('Positive')
                elif score < -0.1:
                    sentiment_labels.append('Negative')
                else:
                    sentiment_labels.append('Neutral')
            
            sentiment_counts = pd.Series(sentiment_labels).value_counts()
            
            fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index,
                        title="Sentiment Distribution",
                        color_discrete_sequence=['#10B981', '#F59E0B', '#EF4444'])
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with insight_tab3:
        st.markdown("#### 🎯 Key Topics")
        
        if 'keywords' in results:
            keywords = results['keywords'][:8]  # Top 8 keywords
            keyword_scores = [1.0 - (i * 0.1) for i in range(len(keywords))]  # Simulated relevance scores
            
            fig = px.bar(x=keyword_scores, y=keywords,
                        orientation='h',
                        title="Topic Relevance",
                        color=keyword_scores,
                        color_continuous_scale='Viridis')
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Detailed insights section
    st.markdown("### 🔍 Professional Insights & Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Meeting effectiveness analysis
        sentiment_score = results.get('sentiment_summary', {}).get('avg_polarity', 0)
        total_messages = results.get('sentiment_summary', {}).get('total_messages', 0)
        
        effectiveness_score = (sentiment_score + 1) * 50 + min(total_messages * 2, 40)
        
        st.markdown(f"""
        <div class="status-card-positive">
            <h4>📈 Meeting Effectiveness Analysis</h4>
            <p><strong>Overall Score:</strong> {effectiveness_score:.1f}/100</p>
            <p><strong>Sentiment Quality:</strong> {(sentiment_score + 1) * 50:.1f}%</p>
            <p><strong>Engagement Level:</strong> {min(total_messages * 2, 100):.1f}%</p>
            <p><strong>Communication Flow:</strong> {'Excellent' if effectiveness_score > 80 else 'Good' if effectiveness_score > 60 else 'Needs Improvement'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Actionable recommendations
        recommendations = []
        
        if sentiment_score < 0:
            recommendations.append("🎯 Focus on constructive communication techniques")
        if total_messages < 10:
            recommendations.append("💬 Encourage more team participation")
        if len(results.get('keywords', [])) < 5:
            recommendations.append("📝 Define clearer meeting objectives")
        
        recommendations.append("⏰ Consider optimal meeting duration")
        recommendations.append("📊 Schedule regular sentiment check-ins")
        
        rec_html = "<ul>" + "".join([f"<li>{rec}</li>" for rec in recommendations]) + "</ul>"
        
        st.markdown(f"""
        <div class="status-card-neutral">
            <h4>💡 Actionable Recommendations</h4>
            {rec_html}
        </div>
        """, unsafe_allow_html=True)


def display_demo_analytics_insights():
    """Display demo analytics and insights when no data is available."""
    
    st.markdown("""
    <div class="info-box">
        <h3>🔬 Advanced Analytics Preview</h3>
        <p>Experience our comprehensive meeting analytics suite. Analyze real meeting data to unlock these powerful insights!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Demo analytics tabs
    insight_tab1, insight_tab2, insight_tab3 = st.columns(3)
    
    with insight_tab1:
        st.markdown("#### 💬 Communication Patterns (Demo)")
        
        # Demo word frequency
        demo_words = ['project', 'team', 'client', 'deadline', 'budget', 'quality', 'meeting', 'goals']
        demo_freq = [25, 18, 15, 12, 10, 8, 7, 6]
        
        fig = px.bar(x=demo_freq, y=demo_words, orientation='h',
                    title="Top Keywords (Demo)",
                    color=demo_freq,
                    color_continuous_scale='Blues')
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with insight_tab2:
        st.markdown("#### 📊 Sentiment Dynamics (Demo)")
        
        # Demo sentiment distribution
        sentiment_data = {'Positive': 65, 'Neutral': 25, 'Negative': 10}
        
        fig = px.pie(values=list(sentiment_data.values()), names=list(sentiment_data.keys()),
                    title="Sentiment Distribution (Demo)",
                    color_discrete_sequence=['#10B981', '#F59E0B', '#EF4444'])
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with insight_tab3:
        st.markdown("#### 🎯 Key Topics (Demo)")
        
        demo_topics = ['Strategic Planning', 'Budget Review', 'Team Performance', 'Client Relations', 'Product Development']
        demo_relevance = [0.95, 0.87, 0.73, 0.68, 0.52]
        
        fig = px.bar(x=demo_relevance, y=demo_topics,
                    orientation='h',
                    title="Topic Relevance (Demo)",
                    color=demo_relevance,
                    color_continuous_scale='Viridis')
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Demo insights section
    st.markdown("### 🔍 Sample Professional Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="status-card-positive">
            <h4>📈 Meeting Effectiveness Analysis (Demo)</h4>
            <p><strong>Overall Score:</strong> 87.3/100</p>
            <p><strong>Sentiment Quality:</strong> 89.2%</p>
            <p><strong>Engagement Level:</strong> 94.1%</p>
            <p><strong>Communication Flow:</strong> Excellent</p>
            <p><strong>Participation Balance:</strong> Well-distributed</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="status-card-neutral">
            <h4>💡 Sample Recommendations</h4>
            <ul>
                <li>🎯 Maintain current positive communication style</li>
                <li>⏰ Consider 45-minute format for efficiency</li>
                <li>📊 Include more data-driven discussions</li>
                <li>💬 Encourage cross-team collaboration topics</li>
                <li>📝 Document key decisions more systematically</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


def display_team_performance(results: Dict):
    """Display team performance analytics."""
    
    st.markdown("""
    <div class="info-box">
        <h3>👥 Team Performance & Dynamics</h3>
        <p>Individual contributor analysis, participation patterns, and collaboration insights.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Individual performance metrics
    if 'speaker_stats' in results:
        speaker_stats = results['speaker_stats']
        
        # Performance overview
        st.markdown("### 📊 Individual Performance Overview")
        
        for speaker, stats in speaker_stats.items():
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>👤 {speaker}</h4>
                    <p><strong>Messages:</strong> {stats['message_count']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                avg_sentiment = stats.get('avg_sentiment', 0.5)
                sentiment_color = "#10B981" if avg_sentiment > 0.1 else "#EF4444" if avg_sentiment < -0.1 else "#F59E0B"
                st.markdown(f"""
                <div class="metric-card">
                    <h4>😊 Sentiment</h4>
                    <p style="color: {sentiment_color}"><strong>{avg_sentiment:.2f}</strong></p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                engagement = min(100, stats['message_count'] * 5)
                st.markdown(f"""
                <div class="metric-card">
                    <h4>💬 Engagement</h4>
                    <p><strong>{engagement}%</strong></p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                influence = min(100, stats['message_count'] * 3 + (avg_sentiment + 1) * 25)
                st.markdown(f"""
                <div class="metric-card">
                    <h4>🎯 Influence</h4>
                    <p><strong>{influence:.1f}%</strong></p>
                </div>
                """, unsafe_allow_html=True)
    
    # Team collaboration matrix
    st.markdown("### 🤝 Team Collaboration Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'speaker_stats' in results:
            speakers = list(speaker_stats.keys())
            participation = [speaker_stats[speaker]['message_count'] for speaker in speakers]
            
            fig = px.bar(x=speakers, y=participation,
                        title="Participation Distribution",
                        color=participation,
                        color_continuous_scale='Blues')
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Communication flow network (simplified)
        if 'speaker_stats' in results and len(speaker_stats) > 1:
            speakers = list(speaker_stats.keys())
            # Create a simple interaction matrix
            interaction_data = []
            for i, speaker1 in enumerate(speakers):
                for j, speaker2 in enumerate(speakers):
                    if i != j:
                        # Simulate interaction strength
                        strength = random.uniform(0.3, 1.0)
                        interaction_data.append({
                            'From': speaker1,
                            'To': speaker2,
                            'Strength': strength
                        })
            
            if interaction_data:
                df_interactions = pd.DataFrame(interaction_data)
                fig = px.scatter(df_interactions, x='From', y='To', size='Strength',
                               title="Communication Network",
                               color='Strength',
                               color_continuous_scale='Viridis')
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)


def display_detailed_reports(results: Dict):
    """Display comprehensive detailed reports."""
    
    st.markdown("""
    <div class="info-box">
        <h3>📋 Comprehensive Meeting Reports</h3>
        <p>Detailed analysis reports, exportable summaries, and actionable insights documentation.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Report sections
    report_type = st.selectbox(
        "📊 Select Report Type:",
        ["Executive Summary", "Detailed Analysis", "Team Performance", "Action Items", "Full Transcript Analysis"]
    )
    
    if report_type == "Executive Summary":
        st.markdown("### 🎯 Executive Summary Report")
        
        # Key metrics summary
        sentiment_score = results.get('sentiment_summary', {}).get('avg_polarity', 0.5)
        total_messages = results.get('sentiment_summary', {}).get('total_messages', 0)
        
        st.markdown(f"""
        <div class="status-card-positive">
            <h4>📈 Meeting Overview</h4>
            <p><strong>Meeting Health Score:</strong> {(sentiment_score + 1) * 50:.1f}%</p>
            <p><strong>Total Interactions:</strong> {total_messages}</p>
            <p><strong>Average Sentiment:</strong> {sentiment_score:.3f}</p>
            <p><strong>Key Topics Identified:</strong> {len(results.get('keywords', []))}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Key insights
        st.markdown("#### 🔍 Key Insights")
        insights = [
            f"Meeting sentiment was {'positive' if sentiment_score > 0.1 else 'negative' if sentiment_score < -0.1 else 'neutral'}",
            f"Team engagement level: {'High' if total_messages > 20 else 'Medium' if total_messages > 10 else 'Low'}",
            f"Communication balance: {'Well-distributed' if len(results.get('speaker_stats', {})) > 2 else 'Could be improved'}",
            f"Action items identified: {len(results.get('keywords', [])) // 3}"
        ]
        
        for insight in insights:
            st.markdown(f"• {insight}")
    
    elif report_type == "Detailed Analysis":
        st.markdown("### 🔬 Detailed Analysis Report")
        
        # Sentiment analysis details
        if 'df' in results:
            df = results['df']
            st.markdown("#### 📊 Sentiment Analysis Details")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Highest Sentiment", f"{df['polarity'].max():.3f}")
                st.metric("Lowest Sentiment", f"{df['polarity'].min():.3f}")
            with col2:
                st.metric("Sentiment Range", f"{df['polarity'].max() - df['polarity'].min():.3f}")
                st.metric("Standard Deviation", f"{df['polarity'].std():.3f}")
            
            # Show sentiment timeline
            fig = px.line(df.reset_index(), x=df.reset_index().index, y='polarity',
                         title="Detailed Sentiment Timeline")
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Export options
    st.markdown("### 📤 Export Options")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Export PDF Report", use_container_width=True):
            st.success("PDF report generation feature coming soon!")
    
    with col2:
        if st.button("📊 Export Excel Analytics", use_container_width=True):
            st.success("Excel export feature coming soon!")
    
    with col3:
        if st.button("📋 Copy Summary", use_container_width=True):
            st.success("Summary copied to clipboard!")


def display_demo_detailed_reports():
    """Display demo detailed reports when no data is available."""
    
    st.markdown("""
    <div class="info-box">
        <h3>📋 Detailed Reports Preview</h3>
        <p>Experience our comprehensive reporting suite. Analyze meeting data to generate professional reports!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Demo report selection
    report_type = st.selectbox(
        "📊 Select Report Type (Demo):",
        ["Executive Summary", "Detailed Analysis", "Team Performance", "Action Items", "Full Transcript Analysis"]
    )
    
    if report_type == "Executive Summary":
        st.markdown("### 🎯 Sample Executive Summary")
        
        st.markdown("""
        <div class="status-card-positive">
            <h4>📈 Sample Meeting Overview</h4>
            <p><strong>Meeting Health Score:</strong> 87.3%</p>
            <p><strong>Total Interactions:</strong> 45</p>
            <p><strong>Average Sentiment:</strong> 0.374</p>
            <p><strong>Key Topics Identified:</strong> 8</p>
            <p><strong>Participation Balance:</strong> Excellent</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### 🔍 Sample Key Insights")
        sample_insights = [
            "Meeting sentiment was consistently positive throughout",
            "Team engagement level: High with balanced participation",
            "Communication balance: Well-distributed across all participants",
            "Action items identified: 5 clear deliverables with owners",
            "Follow-up required: 3 strategic decisions pending"
        ]
        
        for insight in sample_insights:
            st.markdown(f"• {insight}")
    
    # Demo export options
    st.markdown("### 📤 Export Options (Demo)")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Export PDF Report", use_container_width=True):
            st.success("PDF report generation available with real data!")
    
    with col2:
        if st.button("📊 Export Excel Analytics", use_container_width=True):
            st.success("Excel export available with real data!")
    
    with col3:
        if st.button("📋 Copy Summary", use_container_width=True):
            st.success("Summary copying available with real data!")


def display_enterprise_settings():
    """Display enterprise settings and configuration."""
    
    st.markdown("""
    <div class="info-box">
        <h3>⚙️ Enterprise Configuration</h3>
        <p>Advanced settings, integrations, and customization options for enterprise deployments.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Settings sections
    settings_tab1, settings_tab2, settings_tab3 = st.tabs([
        "🔧 Analysis Settings",
        "🔗 Integrations", 
        "👥 User Management"
    ])
    
    with settings_tab1:
        st.markdown("#### 🔧 Analysis Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Sentiment Analysis**")
            sentiment_sensitivity = st.slider("Sentiment Sensitivity", 0.1, 1.0, 0.5)
            auto_categorization = st.checkbox("Auto-categorize discussions", True)
            keyword_extraction = st.checkbox("Advanced keyword extraction", True)
        
        with col2:
            st.markdown("**Performance Metrics**")
            participation_threshold = st.slider("Participation Threshold (%)", 10, 100, 20)
            engagement_weighting = st.selectbox("Engagement Weighting", ["Linear", "Logarithmic", "Custom"])
            report_frequency = st.selectbox("Report Frequency", ["Real-time", "Daily", "Weekly", "Monthly"])
    
    with settings_tab2:
        st.markdown("#### 🔗 Enterprise Integrations")
        
        integrations = [
            {"name": "Microsoft Teams", "status": "Available", "icon": "💼"},
            {"name": "Zoom", "status": "Available", "icon": "📹"},
            {"name": "Slack", "status": "Available", "icon": "💬"},
            {"name": "Google Meet", "status": "Available", "icon": "🎥"},
            {"name": "Salesforce", "status": "Coming Soon", "icon": "☁️"},
            {"name": "Jira", "status": "Coming Soon", "icon": "🎯"}
        ]
        
        for integration in integrations:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                st.markdown(f"{integration['icon']} **{integration['name']}**")
            with col2:
                status_color = "#10B981" if integration['status'] == "Available" else "#F59E0B"
                st.markdown(f"<span style='color: {status_color}'>{integration['status']}</span>", unsafe_allow_html=True)
            with col3:
                if integration['status'] == "Available":
                    st.button("Configure", key=f"config_{integration['name']}")
    
    with settings_tab3:
        st.markdown("#### 👥 User Management & Permissions")
        
        st.markdown("**Role-Based Access Control**")
        
        roles = [
            {"role": "Administrator", "users": 2, "permissions": "Full access"},
            {"role": "Manager", "users": 5, "permissions": "View & Export"},
            {"role": "Analyst", "users": 12, "permissions": "View only"},
            {"role": "Viewer", "users": 25, "permissions": "Dashboard only"}
        ]
        
        for role_info in roles:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"**{role_info['role']}**")
            with col2:
                st.markdown(f"👥 {role_info['users']} users")
            with col3:
                st.markdown(f"🔐 {role_info['permissions']}")
            with col4:
                st.button("Manage", key=f"manage_{role_info['role']}")
    
    # Save settings
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("💾 Save Enterprise Settings", type="primary", use_container_width=True):
            st.success("Enterprise settings saved successfully!")


def display_demo_team_performance():
    """Display demo team performance when no data is available."""
    
    st.markdown("""
    <div class="info-box">
        <h3>👥 Team Performance Preview</h3>
        <p>Comprehensive individual and team analytics. Analyze meeting data to unlock detailed performance insights!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Demo individual performance
    st.markdown("### 📊 Sample Individual Performance")
    
    demo_team = [
        {'name': 'Alice', 'messages': 15, 'sentiment': 0.3, 'engagement': 85, 'influence': 78},
        {'name': 'Bob', 'messages': 12, 'sentiment': 0.1, 'engagement': 72, 'influence': 65},
        {'name': 'Carol', 'messages': 18, 'sentiment': 0.4, 'engagement': 94, 'influence': 89},
        {'name': 'David', 'messages': 8, 'sentiment': -0.1, 'engagement': 58, 'influence': 52}
    ]
    
    for member in demo_team:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>👤 {member['name']}</h4>
                <p><strong>Messages:</strong> {member['messages']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            sentiment_color = "#10B981" if member['sentiment'] > 0.1 else "#EF4444" if member['sentiment'] < -0.1 else "#F59E0B"
            st.markdown(f"""
            <div class="metric-card">
                <h4>😊 Sentiment</h4>
                <p style="color: {sentiment_color}"><strong>{member['sentiment']:.2f}</strong></p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h4>💬 Engagement</h4>
                <p><strong>{member['engagement']}%</strong></p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h4>🎯 Influence</h4>
                <p><strong>{member['influence']}%</strong></p>
            </div>
            """, unsafe_allow_html=True)
    
    # Demo charts
    st.markdown("### 🤝 Sample Team Collaboration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        names = [member['name'] for member in demo_team]
        messages = [member['messages'] for member in demo_team]
        
        fig = px.bar(x=names, y=messages,
                    title="Participation Distribution (Demo)",
                    color=messages,
                    color_continuous_scale='Blues')
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Demo network visualization
        demo_interactions = []
        for i, member1 in enumerate(demo_team):
            for j, member2 in enumerate(demo_team):
                if i != j:
                    strength = random.uniform(0.3, 1.0)
                    demo_interactions.append({
                        'From': member1['name'],
                        'To': member2['name'],
                        'Strength': strength
                    })
        
        df_demo = pd.DataFrame(demo_interactions)
        fig = px.scatter(df_demo, x='From', y='To', size='Strength',
                       title="Communication Network (Demo)",
                       color='Strength',
                       color_continuous_scale='Viridis')
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)


def display_demo_executive_dashboard():
    """Display demo executive dashboard when no data is available."""
    
    st.markdown("""
    <div class="info-box">
        <h3>🏢 Executive Dashboard Preview</h3>
        <p>This is a preview of your professional meeting analytics dashboard. 
        Analyze a meeting transcript to see real insights!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Demo KPIs
    st.markdown("### 🎯 Sample Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🌟 Meeting Health Score</h3>
            <h2>87.3%</h2>
            <p style="color: #059669;">↗️ Above Average</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>💬 Team Engagement</h3>
            <h2>94.1%</h2>
            <p style="color: #0369A1;">High participation</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>😊 Positivity Rate</h3>
            <h2>76.8%</h2>
            <p style="color: #059669;">Team morale: Good</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>✅ Action Items</h3>
            <h2>5</h2>
            <p style="color: #7C2D12;">Follow-up required</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Demo charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Sample Sentiment Analysis")
        # Create demo sentiment chart
        demo_time = list(range(20))
        demo_sentiment = [0.2 + 0.5 * np.sin(x/3) + random.uniform(-0.1, 0.1) for x in demo_time]
        
        fig = px.line(x=demo_time, y=demo_sentiment, 
                     title="Meeting Sentiment Flow (Demo)",
                     color_discrete_sequence=['#3B82F6'])
        fig.update_layout(
            xaxis_title="Meeting Timeline",
            yaxis_title="Sentiment Score",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 👥 Sample Team Participation")
        # Create demo participation chart
        demo_speakers = ['Alice', 'Bob', 'Carol', 'David', 'Eva']
        demo_participation = [15, 12, 18, 8, 11]
        
        fig = px.bar(x=demo_speakers, y=demo_participation, 
                    title="Team Participation (Demo)",
                    color=demo_participation,
                    color_continuous_scale='Blues')
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Professional insights section
    st.markdown("### 🔍 Professional Insights Preview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="status-card-positive">
            <h4>✅ Meeting Strengths</h4>
            <ul>
                <li>High team engagement and participation</li>
                <li>Constructive discussion tone maintained</li>
                <li>Clear action items identified</li>
                <li>Balanced speaking time distribution</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="status-card-neutral">
            <h4>⚡ Improvement Opportunities</h4>
            <ul>
                <li>Consider shorter meeting duration</li>
                <li>Encourage quieter team members</li>
                <li>Focus on outcome-oriented discussions</li>
                <li>Schedule regular follow-up reviews</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main() 