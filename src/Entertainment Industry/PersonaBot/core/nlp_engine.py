"""
NLP Engine for PersonaBot
Handles text generation with personality-aware responses
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import random
import re

# Try to import transformers, but fall back gracefully if not available
try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Transformers library not available: {e}")
    TRANSFORMERS_AVAILABLE = False
    GPT2LMHeadModel = None
    GPT2Tokenizer = None
    GPT2Config = None

from config.settings import settings
from core.personality import PersonalityEncoder

logger = logging.getLogger(__name__)

class NLPEngine:
    """Advanced NLP engine with personality-aware text generation"""
    
    def __init__(self, model_name: Optional[str] = None):
        """Initialize NLP engine"""
        self.model_name = model_name or settings.model.model_name
        self.config = settings.model
        
        # Initialize model and tokenizer
        self._load_model()
        
        # Personality encoder
        self.personality_encoder = PersonalityEncoder()
        
        # Response templates for different personality traits
        self.response_templates = self._initialize_templates()
        
        # Conversation context
        self.conversation_history: List[Dict[str, str]] = []
        
        logger.info(f"NLP Engine initialized with model: {self.model_name}")
    
    def _load_model(self):
        """Load the language model and tokenizer"""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers library not available, using fallback mode")
            self._load_fallback_model()
            return
            
        try:
            # Load tokenizer
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Move to GPU if available
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # Fallback to a simpler approach
            self._load_fallback_model()
    
    def _load_fallback_model(self):
        """Load a fallback model or use template-based responses"""
        logger.warning("Using fallback template-based responses")
        self.model = None
        self.tokenizer = None
        self.device = None
    
    def _initialize_templates(self) -> Dict[str, List[str]]:
        """Initialize response templates for different personality traits"""
        return {
            'humor': [
                "That's hilarious! 😄",
                "Haha, good one!",
                "You've got a great sense of humor!",
                "That made me laugh! 😂",
                "Classic! Love it!"
            ],
            'formal': [
                "I understand your perspective.",
                "That's an interesting point to consider.",
                "I appreciate you sharing that with me.",
                "That's quite insightful.",
                "Thank you for that information."
            ],
            'casual': [
                "Cool!",
                "That's awesome!",
                "Nice one!",
                "Sweet!",
                "That's pretty neat!"
            ],
            'empathetic': [
                "I can see how you'd feel that way.",
                "That sounds really challenging.",
                "I understand what you're going through.",
                "That must be difficult.",
                "I'm here to listen."
            ],
            'enthusiastic': [
                "That's absolutely fantastic!",
                "Wow, that's incredible!",
                "I'm so excited about this!",
                "This is amazing!",
                "I love this!"
            ],
            'professional': [
                "Based on the information provided,",
                "From a professional standpoint,",
                "In my analysis,",
                "Considering the circumstances,",
                "From my perspective,"
            ],
            'creative': [
                "That's such an interesting way to look at it!",
                "I never thought of it that way before!",
                "What a unique perspective!",
                "That's really creative thinking!",
                "I love how you approach this!"
            ],
            'assertive': [
                "I believe that",
                "In my opinion,",
                "I think we should",
                "I'm confident that",
                "I strongly feel that"
            ]
        }
    
    def generate_response(self, 
                         user_message: str,
                         personality_vector: Optional[np.ndarray] = None,
                         context: str = "") -> str:
        """
        Generate a personality-aware response
        
        Args:
            user_message: User's input message
            personality_vector: Current personality vector
            context: Additional context for generation
            
        Returns:
            Generated response string
        """
        if personality_vector is None:
            personality_vector = self.personality_encoder.get_personality_vector()
        
        # Update conversation history
        self.conversation_history.append({
            'role': 'user',
            'content': user_message
        })
        
        # Generate personality-aware prompt
        personality_prompt = self.personality_encoder.get_personality_prompt(context)
        
        # Generate response
        if self.model is not None:
            response = self._generate_with_model(user_message, personality_prompt, personality_vector)
        else:
            response = self._generate_with_templates(user_message, personality_vector)
        
        # Update conversation history
        self.conversation_history.append({
            'role': 'assistant',
            'content': response
        })
        
        # Limit conversation history
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
        
        return response
    
    def _generate_with_model(self, 
                           user_message: str,
                           personality_prompt: str,
                           personality_vector: np.ndarray) -> str:
        """Generate response using the language model"""
        try:
            # Create input prompt
            full_prompt = self._create_model_prompt(user_message, personality_prompt)
            
            # Tokenize input
            inputs = self.tokenizer.encode(full_prompt, return_tensors='pt', truncation=True, max_length=512)
            inputs = inputs.to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 50,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                    repetition_penalty=self.config.repetition_penalty,
                    do_sample=self.config.do_sample,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the new part (after the prompt)
            response = generated_text[len(full_prompt):].strip()
            
            # Clean up response
            response = self._clean_response(response)
            
            # Apply personality adjustments
            response = self._apply_personality_adjustments(response, personality_vector)
            
            return response if response else "I understand what you're saying."
            
        except Exception as e:
            logger.error(f"Error in model generation: {e}")
            return self._generate_with_templates(user_message, personality_vector)
    
    def _generate_with_templates(self, 
                               user_message: str,
                               personality_vector: np.ndarray) -> str:
        """Generate response using templates"""
        # Get dominant personality traits
        traits = settings.personality.traits
        dominant_traits = []
        
        for i, trait in enumerate(traits):
            if personality_vector[i] > 0.7:
                dominant_traits.append(trait)
            elif personality_vector[i] < 0.3:
                dominant_traits.append(f"low_{trait}")
        
        # Select template based on personality
        if dominant_traits:
            # Map traits to template categories
            trait_mapping = {
                'humor': 'humor',
                'formality': 'formal',
                'empathy': 'empathetic',
                'enthusiasm': 'enthusiastic',
                'professionalism': 'professional',
                'creativity': 'creative',
                'assertiveness': 'assertive'
            }
            
            template_category = None
            for trait in dominant_traits:
                if trait in trait_mapping:
                    template_category = trait_mapping[trait]
                    break
            
            if template_category and template_category in self.response_templates:
                templates = self.response_templates[template_category]
                base_response = random.choice(templates)
            else:
                base_response = "That's interesting!"
        else:
            base_response = "I see what you mean."
        
        # Add context-specific response
        response = self._add_context_response(user_message, base_response)
        
        return response
    
    def _create_model_prompt(self, user_message: str, personality_prompt: str) -> str:
        """Create a prompt for the language model"""
        # Build conversation context
        context_messages = []
        for msg in self.conversation_history[-6:]:  # Last 6 messages
            role = "User" if msg['role'] == 'user' else "Assistant"
            context_messages.append(f"{role}: {msg['content']}")
        
        context_str = "\n".join(context_messages)
        
        # Create full prompt
        full_prompt = f"""{personality_prompt}

{context_str}
User: {user_message}
Assistant:"""
        
        return full_prompt
    
    def _clean_response(self, response: str) -> str:
        """Clean up generated response"""
        # Remove extra whitespace
        response = re.sub(r'\s+', ' ', response).strip()
        
        # Remove incomplete sentences at the end
        response = re.sub(r'\s+[A-Z][^.!?]*$', '', response)
        
        # Ensure proper ending
        if response and not response[-1] in '.!?':
            response += '.'
        
        # Limit length
        if len(response.split()) > 50:
            words = response.split()[:50]
            response = ' '.join(words)
            if not response[-1] in '.!?':
                response += '...'
        
        return response
    
    def _apply_personality_adjustments(self, response: str, personality_vector: np.ndarray) -> str:
        """Apply personality-specific adjustments to response"""
        traits = settings.personality.traits
        
        # Add emojis for high humor
        humor_score = personality_vector[traits.index('humor')]
        if humor_score > 0.7:
            emojis = ['😄', '😂', '😊', '😆', '🤣']
            if not any(emoji in response for emoji in emojis):
                response += f" {random.choice(emojis)}"
        
        # Add enthusiasm for high enthusiasm
        enthusiasm_score = personality_vector[traits.index('enthusiasm')]
        if enthusiasm_score > 0.7:
            if not any(word in response.lower() for word in ['!', 'amazing', 'incredible', 'fantastic']):
                response = response.replace('.', '!')
        
        # Add formality for high formality
        formality_score = personality_vector[traits.index('formality')]
        if formality_score > 0.7:
            if not any(word in response.lower() for word in ['indeed', 'furthermore', 'moreover']):
                response = f"Indeed, {response.lower()}"
        
        return response
    
    def _add_context_response(self, user_message: str, base_response: str) -> str:
        """Add context-specific response elements"""
        user_lower = user_message.lower()
        
        # Question detection
        if '?' in user_message:
            if 'how' in user_lower:
                return f"I'd be happy to explain how that works. {base_response}"
            elif 'what' in user_lower:
                return f"That's a great question. {base_response}"
            elif 'why' in user_lower:
                return f"Let me think about that. {base_response}"
            else:
                return f"Good question! {base_response}"
        
        # Greeting detection
        greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']
        if any(greeting in user_lower for greeting in greetings):
            return f"Hello! {base_response}"
        
        # Agreement detection
        agreements = ['yes', 'yeah', 'sure', 'okay', 'alright', 'absolutely']
        if any(agreement in user_lower for agreement in agreements):
            return f"I agree! {base_response}"
        
        # Disagreement detection
        disagreements = ['no', 'nope', 'not really', 'disagree']
        if any(disagreement in user_lower for disagreement in disagreements):
            return f"I see your point. {base_response}"
        
        return base_response
    
    def get_conversation_context(self) -> str:
        """Get current conversation context"""
        if not self.conversation_history:
            return ""
        
        # Get last few messages for context
        recent_messages = self.conversation_history[-4:]
        context_parts = []
        
        for msg in recent_messages:
            role = "User" if msg['role'] == 'user' else "Assistant"
            content = msg['content'][:100]  # Limit length
            context_parts.append(f"{role}: {content}")
        
        return "\n".join(context_parts)
    
    def reset_conversation(self):
        """Reset conversation history"""
        self.conversation_history = []
        logger.info("Conversation history reset")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if self.model is None:
            return {
                "model_type": "template-based",
                "model_name": "fallback",
                "device": "cpu",
                "parameters": 0
            }
        
        return {
            "model_type": "transformer",
            "model_name": self.model_name,
            "device": str(self.device),
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "config": {
                "max_length": self.config.max_length,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k
            }
        }
