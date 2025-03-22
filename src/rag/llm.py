"""
LLM interface using Ollama for the RAG pipeline
"""
import json
from typing import Dict, Any, Optional, List
import requests

from config.config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    SYSTEM_PROMPT,
    TEMPERATURE
)


class LLMInterface:
    """
    Interface for Large Language Model using Ollama
    """
    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model_name: str = OLLAMA_MODEL,
        system_prompt: str = SYSTEM_PROMPT,
        temperature: float = TEMPERATURE
    ):
        """
        Initialize the LLM interface
        
        Args:
            base_url: Base URL for Ollama API
            model_name: Name of the model to use
            system_prompt: System prompt for the LLM
            temperature: Temperature for generation (higher = more random)
        """
        self.base_url = base_url
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.temperature = temperature
        
        # Ensure base_url doesn't end with slash
        if self.base_url.endswith("/"):
            self.base_url = self.base_url[:-1]
    
    def generate(
        self,
        prompt: str,
        context: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Generate a response using the LLM
        
        Args:
            prompt: User prompt
            context: Optional context from retrieval
            temperature: Optional temperature override
            
        Returns:
            Generated text response
        """
        # Build the full prompt
        full_prompt = self._build_prompt(prompt, context)
        
        # Prepare request
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "system": self.system_prompt,
            "temperature": temperature if temperature is not None else self.temperature,
            "stream": False
        }
        
        # Make the request
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            
            # Parse the response
            result = response.json()
            return result.get("response", "")
        
        except requests.exceptions.RequestException as e:
            print(f"Error calling Ollama API: {e}")
            return f"Error: Could not generate response. Please check if Ollama is running. Details: {e}"
    
    def _build_prompt(self, prompt: str, context: Optional[str] = None) -> str:
        """
        Build the full prompt with context
        
        Args:
            prompt: User prompt
            context: Optional context from retrieval
            
        Returns:
            Full prompt for the LLM
        """
        if context:
            return (
                "You are an assistant specialized in Scottish Parliament proceedings. "
                "Answer the following question based only on the provided context. "
                "If you can't find the answer in the context, say that you don't have "
                "enough information.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {prompt}\n\n"
                "Answer:"
            )
        else:
            return (
                "You are an assistant specialized in Scottish Parliament proceedings. "
                "The following question has no context provided, so please respond "
                "that you need more specific information about parliamentary "
                "proceedings to provide a proper answer.\n\n"
                f"Question: {prompt}\n\n"
                "Answer:"
            )
    
    def generate_structured_output(
        self,
        prompt: str,
        context: Optional[str] = None,
        output_schema: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Generate a structured JSON output using the LLM
        
        Args:
            prompt: User prompt
            context: Optional context from retrieval
            output_schema: JSON schema for the output
            
        Returns:
            Generated structured output
        """
        # If no schema is provided, use a default one
        if output_schema is None:
            output_schema = {
                "answer": "string",
                "sources": ["string"],
                "confidence": "number between 0 and 1"
            }
        
        # Build prompt with instructions for structured output
        structured_prompt = (
            f"Based on the following context, answer the question and return a JSON object "
            f"with the following structure: {json.dumps(output_schema)}\n\n"
        )
        
        if context:
            structured_prompt += f"Context:\n{context}\n\n"
        
        structured_prompt += f"Question: {prompt}\n\nJSON Response:"
        
        # Generate response
        response = self.generate(structured_prompt, temperature=0.1)
        
        # Try to parse JSON from the response
        try:
            # Find JSON in the response (it might be wrapped in markdown code blocks or other text)
            json_str = self._extract_json(response)
            
            # Parse the JSON
            result = json.loads(json_str)
            return result
        except Exception as e:
            print(f"Error parsing structured output: {e}")
            return {
                "answer": response,
                "sources": [],
                "confidence": 0.0,
                "error": "Failed to parse structured output"
            }
    
    def _extract_json(self, text: str) -> str:
        """
        Extract JSON from a text that might contain markdown or other formatting
        
        Args:
            text: Text to extract JSON from
            
        Returns:
            Extracted JSON string
        """
        # Try to find JSON blocks
        if "```json" in text:
            # Extract text between ```json and ```
            parts = text.split("```json")
            if len(parts) > 1:
                json_part = parts[1].split("```")[0].strip()
                return json_part
        
        # If no JSON block, try to find anything that looks like a JSON object
        if "{" in text and "}" in text:
            start = text.find("{")
            end = text.rfind("}") + 1
            return text[start:end]
        
        # If all else fails, return the original text
        return text 