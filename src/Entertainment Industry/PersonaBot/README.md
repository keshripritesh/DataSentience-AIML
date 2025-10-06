# 🤖 **PersonaBot** - Advanced Adaptive Conversational AI

> **Next-Generation Conversational AI with Dynamic Personality Adaptation using Reinforcement Learning**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)](tests/)

## 🎯 **Project Overview**

PersonaBot is a cutting-edge conversational AI system that leverages **Reinforcement Learning** and **Advanced NLP** to create dynamic, adaptive personalities that evolve based on user interactions. Unlike traditional chatbots with static responses, PersonaBot continuously learns and adapts its communication style to maximize engagement and user satisfaction.

### 🌟 **Key Features**

- **🧠 Dynamic Personality Adaptation**: Real-time personality evolution based on user feedback
- **🎯 Reinforcement Learning Engine**: Advanced RL algorithms for optimal response generation
- **📊 Sentiment-Aware Responses**: Contextual understanding and emotional intelligence
- **🔄 Continuous Learning**: Online learning from every interaction
- **📈 Performance Analytics**: Real-time metrics and personality drift visualization
- **🔧 Modular Architecture**: Extensible design for easy customization

## 🏗️ **Architecture**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Input    │───▶│  NLP Engine     │───▶│  Personality    │
│                 │    │  (Transformer)  │    │  Encoder        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Response       │◀───│  RL Agent       │◀───│  Reward         │
│  Generation     │    │  (Actor-Critic) │    │  Function       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 **Quick Start**

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- Transformers 4.0+

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/personabot.git
cd personabot

# Install dependencies
pip install -r requirements.txt

# Run tests to ensure everything works
python -m pytest tests/ -v

# Start the CLI interface
python ui/cli.py
```

### Web Interface

```bash
# Launch the Streamlit web app
streamlit run ui/web_app.py
```

## 📊 **Usage Examples**

### Basic Conversation
```python
from core.personabot import PersonaBot

bot = PersonaBot()
response = bot.chat("Hello! How are you today?")
print(response)  # Adaptive response based on personality
```

### Personality Customization
```python
# Initialize with specific personality traits
personality_config = {
    "humor": 0.8,
    "formality": 0.2,
    "empathy": 0.9,
    "sarcasm": 0.3
}
bot = PersonaBot(personality_config=personality_config)
```

## 🧪 **Testing**

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_rl_agent.py -v
python -m pytest tests/test_nlp_engine.py -v

# Run with coverage
python -m pytest tests/ --cov=core --cov-report=html
```

## 📁 **Project Structure**

```
personabot/
├── core/                    # Core AI components
│   ├── nlp_engine.py       # Transformer-based NLP engine
│   ├── rl_agent.py         # Reinforcement Learning agent
│   ├── personality.py      # Personality encoding/decoding
│   └── reward.py           # Reward functions & metrics
├── ui/                     # User interfaces
│   ├── cli.py             # Command-line interface
│   └── web_app.py         # Streamlit web application
├── utils/                  # Utility functions
│   ├── sentiment.py       # Sentiment analysis
│   ├── metrics.py         # Performance metrics
│   └── visualization.py   # Personality drift visualization
├── data/                   # Data storage
│   ├── sessions/          # Conversation logs
│   └── models/            # Trained model checkpoints
├── tests/                  # Comprehensive test suite
│   ├── test_nlp_engine.py
│   ├── test_rl_agent.py
│   ├── test_personality.py
│   └── test_integration.py
├── config/                 # Configuration files
│   └── settings.py
├── requirements.txt        # Python dependencies
├── setup.py               # Package setup
└── README.md              # This file
```

## 🔬 **Technical Details**

### NLP Engine
- **Base Model**: GPT-2 Small (117M parameters)
- **Fine-tuning**: Custom personality-aware training
- **Context Window**: 512 tokens with sliding attention

### Reinforcement Learning
- **Algorithm**: Actor-Critic with Advantage Actor-Critic (A2C)
- **State Space**: Conversation history + personality vector
- **Action Space**: Response generation with personality influence
- **Reward Function**: Multi-objective optimization

### Personality Encoding
- **Dimensions**: 8-dimensional personality vector
- **Traits**: Humor, Formality, Empathy, Sarcasm, Enthusiasm, Professionalism, Creativity, Assertiveness
- **Adaptation**: Real-time gradient-based updates

## 📈 **Performance Metrics**

- **Engagement Rate**: Average conversation length
- **Sentiment Score**: User satisfaction tracking
- **Personality Stability**: Consistency in adaptive behavior
- **Response Quality**: Relevance and coherence scores

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- Hugging Face Transformers library
- PyTorch team for the RL framework
- OpenAI for the base GPT-2 model
- Streamlit for the web interface

## 📞 **Contact**

- **Author**: Your Name
- **Email**: your.email@example.com
- **GitHub**: [@yourusername](https://github.com/yourusername)

---

**⭐ Star this repository if you find it useful!**
