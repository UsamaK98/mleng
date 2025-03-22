"""
Interface for LLM models via Ollama
"""
import json
import requests
from typing import Dict, Any, List, Optional

from config.config import OLLAMA_BASE_URL, OLLAMA_MODEL, TEMPERATURE, SYSTEM_PROMPT


class LLMInterface:
    """
    Interface for interacting with LLMs via Ollama
    """
    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = OLLAMA_MODEL,
        system_prompt: str = SYSTEM_PROMPT,
        temperature: float = TEMPERATURE
    ):
        """
        Initialize the LLM interface
        
        Args:
            base_url: Ollama API base URL
            model: Name of the model to use
            system_prompt: System prompt to use
            temperature: Temperature parameter for generation
        """
        self.base_url = base_url
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
    
    def generate(
        self,
        prompt: str,
        context: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Generate a response from the LLM
        
        Args:
            prompt: The prompt to send to the model
            context: Optional context information
            temperature: Optional temperature override
            
        Returns:
            Dictionary with the model's response
        """
        # Use default temperature if not specified
        temp = temperature if temperature is not None else self.temperature
        
        # Prepare the API request to Ollama
        api_url = f"{self.base_url}/api/chat"
        
        # Format the messages
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # Add context messages if provided
        if context:
            for ctx in context:
                messages.append({"role": "user", "content": ctx.get("user", "")})
                if "assistant" in ctx:
                    messages.append({"role": "assistant", "content": ctx["assistant"]})
        
        # Add the current prompt
        messages.append({"role": "user", "content": prompt})
        
        # Prepare the payload
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temp,
            "stream": False
        }
        
        try:
            # Make the request to Ollama
            response = requests.post(api_url, json=payload)
            response.raise_for_status()
            
            # Parse the response
            result = response.json()
            
            # Return the response content
            return {
                "answer": result["message"]["content"],
                "model": self.model,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}  # Ollama doesn't provide token counts
            }
            
        except requests.exceptions.RequestException as e:
            # Handle request errors
            error_msg = f"Error calling Ollama API: {str(e)}"
            print(error_msg)
            return {"answer": error_msg, "model": self.model, "usage": {}}
    
    def generate_with_context(
        self,
        prompt: str,
        context_texts: List[str],
        context_metadata: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Generate a response with provided context documents
        
        Args:
            prompt: The prompt to send to the model
            context_texts: List of context texts to include
            context_metadata: Optional metadata for each context text
            temperature: Optional temperature override
            
        Returns:
            Dictionary with the model's response
        """
        # Format the combined prompt with context
        combined_prompt = self._format_prompt_with_context(prompt, context_texts, context_metadata)
        
        # Generate the response using the combined prompt
        return self.generate(combined_prompt, temperature=temperature)
    
    def _format_prompt_with_context(
        self,
        prompt: str,
        context_texts: List[str],
        context_metadata: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Format the prompt with context information
        
        Args:
            prompt: The user's question
            context_texts: List of relevant context texts
            context_metadata: Optional metadata for each context text
            
        Returns:
            Formatted prompt with context
        """
        # Start with a header
        formatted_prompt = "I'll provide you with some extracts from parliamentary minutes, followed by a question. Please answer the question based on the provided extracts only.\n\n"
        
        # Add each context with its metadata
        for i, text in enumerate(context_texts):
            formatted_prompt += f"--- Extract {i+1} ---\n"
            
            # Add metadata if available
            if context_metadata and i < len(context_metadata):
                metadata = context_metadata[i]
                date = metadata.get("date", "Unknown date")
                speaker = metadata.get("speaker", "Unknown speaker")
                formatted_prompt += f"Date: {date}, Speaker: {speaker}\n"
            
            # Add the text
            formatted_prompt += f"{text}\n\n"
        
        # Add the question
        formatted_prompt += f"--- Question ---\n{prompt}\n\n"
        formatted_prompt += "Please answer the question based only on the information provided in the extracts. If the answer cannot be determined from the extracts, please say so."
        
        return formatted_prompt 