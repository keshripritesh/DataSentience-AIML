"""
Personality encoding and decoding for PersonaBot
Handles personality vector representation and adaptation
"""

import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

from config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class PersonalityState:
    """Represents the current personality state"""
    traits: Dict[str, float]
    timestamp: datetime
    confidence: float = 1.0
    adaptation_rate: float = 0.1
    
    def to_vector(self) -> np.ndarray:
        """Convert personality to vector representation"""
        return np.array([self.traits[trait] for trait in settings.personality.traits])
    
    @classmethod
    def from_vector(cls, vector: np.ndarray, confidence: float = 1.0) -> 'PersonalityState':
        """Create personality state from vector"""
        traits = dict(zip(settings.personality.traits, vector))
        return cls(traits=traits, timestamp=datetime.now(), confidence=confidence)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "traits": self.traits,
            "timestamp": self.timestamp.isoformat(),
            "confidence": self.confidence,
            "adaptation_rate": self.adaptation_rate
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PersonalityState':
        """Create from dictionary"""
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)

class PersonalityEncoder:
    """Handles personality encoding and adaptation"""
    
    def __init__(self, initial_personality: Optional[Dict[str, float]] = None):
        """Initialize personality encoder"""
        self.traits = settings.personality.traits
        self.default_values = settings.personality.default_values
        
        if initial_personality is None:
            initial_personality = self.default_values.copy()
        
        self.current_state = PersonalityState(
            traits=initial_personality,
            timestamp=datetime.now()
        )
        
        self.history: List[PersonalityState] = [self.current_state]
        self.adaptation_history: List[Tuple[datetime, Dict[str, float]]] = []
    
    def get_personality_vector(self) -> np.ndarray:
        """Get current personality as vector"""
        return self.current_state.to_vector()
    
    def get_personality_dict(self) -> Dict[str, float]:
        """Get current personality as dictionary"""
        return self.current_state.traits.copy()
    
    def adapt_personality(self, 
                         feedback: Dict[str, float], 
                         learning_rate: float = 0.1,
                         confidence: float = 1.0) -> None:
        """
        Adapt personality based on feedback
        
        Args:
            feedback: Dictionary mapping trait names to feedback scores (-1 to 1)
            learning_rate: Rate of adaptation
            confidence: Confidence in the feedback (0 to 1)
        """
        old_traits = self.current_state.traits.copy()
        new_traits = {}
        
        for trait in self.traits:
            if trait in feedback:
                # Calculate adaptation
                current_value = old_traits[trait]
                feedback_score = feedback[trait]
                
                # Clamp feedback to valid range
                feedback_score = np.clip(feedback_score, -1.0, 1.0)
                
                # Apply adaptation with momentum
                adaptation = learning_rate * feedback_score * confidence
                new_value = np.clip(current_value + adaptation, 0.0, 1.0)
                
                new_traits[trait] = new_value
            else:
                new_traits[trait] = old_traits[trait]
        
        # Create new personality state
        self.current_state = PersonalityState(
            traits=new_traits,
            timestamp=datetime.now(),
            confidence=confidence,
            adaptation_rate=learning_rate
        )
        
        # Record history
        self.history.append(self.current_state)
        self.adaptation_history.append((
            datetime.now(),
            {k: new_traits[k] - old_traits[k] for k in self.traits}
        ))
        
        logger.info(f"Personality adapted: {old_traits} -> {new_traits}")
    
    def get_personality_prompt(self, context: str = "") -> str:
        """
        Generate personality-aware prompt for the language model
        
        Args:
            context: Additional context to include
            
        Returns:
            Formatted prompt string
        """
        traits = self.current_state.traits
        
        # Create personality description
        personality_desc = []
        
        if traits["humor"] > 0.7:
            personality_desc.append("witty and humorous")
        elif traits["humor"] < 0.3:
            personality_desc.append("serious and focused")
        
        if traits["formality"] > 0.7:
            personality_desc.append("formal and professional")
        elif traits["formality"] < 0.3:
            personality_desc.append("casual and friendly")
        
        if traits["empathy"] > 0.7:
            personality_desc.append("empathetic and understanding")
        elif traits["empathy"] < 0.3:
            personality_desc.append("direct and objective")
        
        if traits["sarcasm"] > 0.7:
            personality_desc.append("sarcastic and witty")
        elif traits["sarcasm"] < 0.3:
            personality_desc.append("sincere and straightforward")
        
        if traits["enthusiasm"] > 0.7:
            personality_desc.append("enthusiastic and energetic")
        elif traits["enthusiasm"] < 0.3:
            personality_desc.append("calm and measured")
        
        if traits["professionalism"] > 0.7:
            personality_desc.append("professional and business-like")
        elif traits["professionalism"] < 0.3:
            personality_desc.append("relaxed and informal")
        
        if traits["creativity"] > 0.7:
            personality_desc.append("creative and imaginative")
        elif traits["creativity"] < 0.3:
            personality_desc.append("practical and logical")
        
        if traits["assertiveness"] > 0.7:
            personality_desc.append("assertive and confident")
        elif traits["assertiveness"] < 0.3:
            personality_desc.append("gentle and accommodating")
        
        # Default personality if no strong traits
        if not personality_desc:
            personality_desc = ["balanced and adaptable"]
        
        personality_str = ", ".join(personality_desc)
        
        prompt = f"""You are an AI assistant with a {personality_str} personality. 
Your communication style reflects these traits naturally in your responses.
{context}

Respond in a way that reflects your personality:"""
        
        return prompt
    
    def get_personality_metrics(self) -> Dict[str, Any]:
        """Get personality metrics and statistics"""
        if len(self.history) < 2:
            return {"stability": 1.0, "adaptation_count": 0, "drift": 0.0}
        
        # Calculate stability (inverse of variance)
        vectors = [state.to_vector() for state in self.history]
        variance = np.var(vectors, axis=0).mean()
        stability = 1.0 / (1.0 + variance)
        
        # Calculate total drift
        initial_vector = vectors[0]
        current_vector = vectors[-1]
        drift = np.linalg.norm(current_vector - initial_vector)
        
        return {
            "stability": float(stability),
            "adaptation_count": len(self.adaptation_history),
            "drift": float(drift),
            "current_traits": self.current_state.traits,
            "history_length": len(self.history)
        }
    
    def save_personality(self, filepath: str) -> None:
        """Save personality state to file"""
        # Convert current state to serializable format
        current_state_dict = self.current_state.to_dict()
        current_state_dict['traits'] = {k: float(v) for k, v in current_state_dict['traits'].items()}
        current_state_dict['confidence'] = float(current_state_dict['confidence'])
        current_state_dict['adaptation_rate'] = float(current_state_dict['adaptation_rate'])
        
        # Convert history to serializable format
        serializable_history = []
        for state in self.history:
            state_dict = state.to_dict()
            state_dict['traits'] = {k: float(v) for k, v in state_dict['traits'].items()}
            state_dict['confidence'] = float(state_dict['confidence'])
            state_dict['adaptation_rate'] = float(state_dict['adaptation_rate'])
            serializable_history.append(state_dict)
        
        # Convert adaptation history to serializable format
        serializable_adaptation_history = []
        for timestamp, adaptations in self.adaptation_history:
            serializable_adaptations = {k: float(v) for k, v in adaptations.items()}
            serializable_adaptation_history.append((timestamp.isoformat(), serializable_adaptations))
        
        data = {
            "current_state": current_state_dict,
            "history": serializable_history,
            "adaptation_history": serializable_adaptation_history
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Personality saved to {filepath}")
    
    def load_personality(self, filepath: str) -> None:
        """Load personality state from file"""
        if not os.path.exists(filepath):
            logger.warning(f"Personality file not found: {filepath}")
            return
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.current_state = PersonalityState.from_dict(data["current_state"])
        self.history = [PersonalityState.from_dict(state) for state in data["history"]]
        self.adaptation_history = [
            (datetime.fromisoformat(timestamp), adaptations)
            for timestamp, adaptations in data["adaptation_history"]
        ]
        
        logger.info(f"Personality loaded from {filepath}")
    
    def reset_personality(self) -> None:
        """Reset personality to default values"""
        self.current_state = PersonalityState(
            traits=self.default_values.copy(),
            timestamp=datetime.now()
        )
        self.history = [self.current_state]
        self.adaptation_history = []
        
        logger.info("Personality reset to default values")
    
    def get_personality_summary(self) -> str:
        """Get a human-readable summary of current personality"""
        traits = self.current_state.traits
        
        summary_parts = []
        for trait, value in traits.items():
            if value > 0.7:
                summary_parts.append(f"High {trait}")
            elif value < 0.3:
                summary_parts.append(f"Low {trait}")
            else:
                summary_parts.append(f"Moderate {trait}")
        
        return f"Personality: {', '.join(summary_parts)}"
