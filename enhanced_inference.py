"""
Enhanced Inference System
Combines trained model with RAG and external knowledge
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from typing import Dict, List, Optional
import json

from post_training_config import (
    BASE_MODEL_DIR, INFERENCE_CONFIG, SYSTEM_PROMPT, 
    RESPONSE_CONFIG, OPTIMIZATION_CONFIG
)
from rag_system import RAGSystem
from knowledge_integration import ExternalKnowledgeIntegration


class EnhancedInference:
    """
    Enhanced inference system with RAG and external knowledge
    """
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path or BASE_MODEL_DIR
        self.model = None
        self.tokenizer = None
        self.rag_system = None
        self.knowledge_integration = None
        
        print("🚀 Initializing Enhanced Inference System...")
        self._load_model()
        self._initialize_systems()
    
    def _load_model(self):
        """Load the trained model"""
        try:
            print(f"\n📥 Loading model from: {self.model_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                "EleutherAI/gpt-j-6B",
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print("✅ Tokenizer loaded")
            
            # Configure quantization
            if OPTIMIZATION_CONFIG["use_8bit"]:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0
                )
            else:
                quantization_config = None
            
            # Load base model
            print("📥 Loading base model (this may take a few minutes)...")
            self.model = AutoModelForCausalLM.from_pretrained(
                "EleutherAI/gpt-j-6B",
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16 if OPTIMIZATION_CONFIG["use_gpu"] else torch.float32
            )
            
            # Load LoRA adapter if exists
            if os.path.exists(self.model_path):
                print(f"📥 Loading LoRA adapter from: {self.model_path}")
                self.model = PeftModel.from_pretrained(self.model, self.model_path)
                print("✅ LoRA adapter loaded")
            else:
                print("⚠️  No trained adapter found, using base model")
            
            self.model.eval()
            print("✅ Model loaded and ready")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def _initialize_systems(self):
        """Initialize RAG and knowledge integration systems"""
        try:
            print("\n🔧 Initializing enhancement systems...")
            
            # Initialize RAG
            self.rag_system = RAGSystem()
            
            # Initialize knowledge integration
            self.knowledge_integration = ExternalKnowledgeIntegration()
            
            print("✅ Enhancement systems initialized")
            
        except Exception as e:
            print(f"⚠️  Failed to initialize enhancement systems: {e}")
            print("⚠️  Continuing with model-only inference")
    
    def _format_prompt(self, query: str, context: str = None) -> str:
        """Format prompt with system message and context"""
        prompt = f"{SYSTEM_PROMPT}\n\n"
        
        if context:
            prompt += f"{context}\n\n"
        
        prompt += f"### User Query:\n{query}\n\n### Response:\n"
        
        return prompt
    
    def _should_use_rag(self, query: str) -> bool:
        """Determine if RAG should be used for this query"""
        # Keywords that suggest need for real-time information
        realtime_keywords = [
            "latest", "recent", "current", "today", "now", "trending",
            "news", "price", "weather", "stock", "crypto"
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in realtime_keywords)
    
    def _should_execute_code(self, query: str) -> bool:
        """Determine if code execution is needed"""
        execution_keywords = [
            "run this", "execute", "test this code", "what does this output",
            "calculate", "compute"
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in execution_keywords)
    
    def generate(
        self,
        query: str,
        use_rag: bool = None,
        use_knowledge: bool = True,
        max_length: int = None,
        temperature: float = None,
        **kwargs
    ) -> Dict[str, any]:
        """
        Generate response with optional RAG and knowledge integration
        
        Args:
            query: User query
            use_rag: Whether to use RAG (auto-detected if None)
            use_knowledge: Whether to use external knowledge
            max_length: Maximum generation length
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
        
        Returns:
            Dictionary with response and metadata
        """
        try:
            # Auto-detect RAG usage
            if use_rag is None:
                use_rag = self._should_use_rag(query)
            
            # Retrieve context if needed
            context = ""
            sources = []
            
            if use_rag and self.rag_system:
                print("🔍 Retrieving real-time information...")
                documents = self.rag_system.retrieve(query)
                context = self.rag_system.format_context(documents)
                sources = [{"source": doc.source, "url": doc.url} for doc in documents]
            
            # Check if code execution is needed
            code_result = None
            if self._should_execute_code(query) and self.knowledge_integration:
                # Extract code from query (simple heuristic)
                if "```" in query:
                    code_start = query.find("```") + 3
                    code_end = query.find("```", code_start)
                    if code_end > code_start:
                        code = query[code_start:code_end].strip()
                        # Detect language
                        first_line = code.split('\n')[0]
                        if first_line in ["python", "bash", "javascript"]:
                            language = first_line
                            code = '\n'.join(code.split('\n')[1:])
                        else:
                            language = "python"
                        
                        print(f"⚙️  Executing {language} code...")
                        code_result = self.knowledge_integration.execute_code(code, language)
            
            # Format prompt
            prompt = self._format_prompt(query, context)
            
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            )
            
            if OPTIMIZATION_CONFIG["use_gpu"]:
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Generate
            generation_config = {
                "max_length": max_length or INFERENCE_CONFIG["max_length"],
                "temperature": temperature or INFERENCE_CONFIG["temperature"],
                "top_p": INFERENCE_CONFIG["top_p"],
                "top_k": INFERENCE_CONFIG["top_k"],
                "repetition_penalty": INFERENCE_CONFIG["repetition_penalty"],
                "do_sample": INFERENCE_CONFIG["do_sample"],
                "num_beams": INFERENCE_CONFIG["num_beams"],
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                **kwargs
            }
            
            print("🤖 Generating response...")
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generation_config
                )
            
            # Decode
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the response part (after "### Response:")
            if "### Response:" in response:
                response = response.split("### Response:")[-1].strip()
            
            # Add code execution result if available
            if code_result:
                response += f"\n\n**Code Execution Result:**\n"
                if code_result.get("success"):
                    response += f"```\n{code_result['stdout']}\n```"
                else:
                    response += f"Error: {code_result.get('error', code_result.get('stderr'))}"
            
            # Store interaction in knowledge base
            if use_knowledge and self.knowledge_integration:
                self.knowledge_integration.store_interaction(
                    query=query,
                    response=response,
                    source="enhanced_model",
                    metadata={
                        "used_rag": use_rag,
                        "num_sources": len(sources),
                        "code_executed": code_result is not None
                    }
                )
            
            # Build result
            result = {
                "query": query,
                "response": response,
                "metadata": {
                    "used_rag": use_rag,
                    "sources": sources if RESPONSE_CONFIG["include_sources"] else [],
                    "code_execution": code_result,
                    "model_path": self.model_path
                }
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            return {
                "query": query,
                "response": f"Error: {str(e)}",
                "metadata": {"error": str(e)}
            }
    
    def chat(self, query: str, **kwargs) -> str:
        """Simple chat interface"""
        result = self.generate(query, **kwargs)
        return result["response"]


# Example usage and testing
if __name__ == "__main__":
    # Initialize system
    print("="*80)
    print("ENHANCED INFERENCE SYSTEM")
    print("="*80)
    
    inference = EnhancedInference()
    
    # Test queries
    test_queries = [
        "What is the current price of Bitcoin?",
        "Write a Python function to implement bubble sort",
        "Explain how neural networks work",
    ]
    
    for query in test_queries:
        print("\n" + "="*80)
        print(f"Query: {query}")
        print("="*80)
        
        result = inference.generate(query)
        
        print(f"\nResponse:\n{result['response']}")
        
        if result['metadata'].get('sources'):
            print(f"\nSources used: {len(result['metadata']['sources'])}")
            for source in result['metadata']['sources'][:3]:
                print(f"  - {source['source']}: {source['url']}")
