"""
PersonaBot Core Package
Advanced conversational AI with dynamic personality adaptation
"""

from .personabot import PersonaBot
from .personality import PersonalityEncoder, PersonalityState
from .nlp_engine import NLPEngine
from .rl_agent import RLAgent, ActorCriticNetwork
from .reward import RewardFunction, ConversationState

__version__ = "1.0.0"
__author__ = "PersonaBot Team"

__all__ = [
    "PersonaBot",
    "PersonalityEncoder", 
    "PersonalityState",
    "NLPEngine",
    "RLAgent",
    "ActorCriticNetwork",
    "RewardFunction",
    "ConversationState"
]
