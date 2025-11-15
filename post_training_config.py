"""
Post-Training Enhancement Configuration
Adds RAG, External Knowledge, and Advanced Capabilities
"""

# Model Configuration
BASE_MODEL_DIR = "./dLNk-gpt-uncensored-v2"  # From V2 training
ENHANCED_MODEL_DIR = "./dLNk-gpt-enhanced"

# RAG Configuration
RAG_CONFIG = {
    "enabled": True,
    "vector_db": "faiss",  # or "chroma"
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "chunk_size": 512,
    "chunk_overlap": 50,
    "top_k": 5,  # Number of relevant documents to retrieve
    "similarity_threshold": 0.7
}

# Knowledge Sources Configuration
KNOWLEDGE_SOURCES = {
    "web_search": {
        "enabled": True,
        "engines": ["duckduckgo", "searx"],  # Privacy-focused search
        "max_results": 10
    },
    "wikipedia": {
        "enabled": True,
        "language": "en"
    },
    "github": {
        "enabled": True,
        "search_code": True,
        "search_repos": True
    },
    "stackoverflow": {
        "enabled": True,
        "tags": ["python", "javascript", "security", "algorithms"]
    }
}

# External APIs Configuration
EXTERNAL_APIS = {
    "code_execution": {
        "enabled": True,
        "sandbox": "docker",  # Safe code execution
        "timeout": 30,
        "languages": ["python", "javascript", "bash"]
    },
    "web_scraping": {
        "enabled": True,
        "user_agent": "dLNk-GPT-Bot/1.0",
        "respect_robots_txt": True
    },
    "database": {
        "enabled": True,
        "type": "sqlite",  # Can be changed to postgresql, mysql
        "path": "./knowledge_base.db"
    }
}

# Advanced Code Generation Configuration
CODE_GEN_CONFIG = {
    "enabled": True,
    "features": {
        "syntax_checking": True,
        "auto_testing": True,
        "documentation_generation": True,
        "code_explanation": True,
        "refactoring_suggestions": True
    },
    "supported_languages": [
        "python", "javascript", "typescript", "java", "c", "cpp",
        "go", "rust", "php", "ruby", "swift", "kotlin", "bash"
    ],
    "code_quality_tools": {
        "python": ["pylint", "black", "mypy"],
        "javascript": ["eslint", "prettier"],
        "general": ["sonarqube"]
    }
}

# Real-time Data Sources
REALTIME_SOURCES = {
    "news_apis": {
        "enabled": True,
        "sources": ["newsapi", "gdelt"],
        "categories": ["technology", "security", "programming"]
    },
    "crypto_prices": {
        "enabled": True,
        "api": "coingecko"
    },
    "weather": {
        "enabled": True,
        "api": "openweathermap"
    },
    "stock_market": {
        "enabled": True,
        "api": "alphavantage"
    }
}

# Caching Configuration
CACHE_CONFIG = {
    "enabled": True,
    "backend": "redis",  # or "memory"
    "ttl": 3600,  # Cache time-to-live in seconds
    "max_size": "1GB"
}

# Inference Configuration
INFERENCE_CONFIG = {
    "max_length": 2048,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1,
    "do_sample": True,
    "num_beams": 1,
    "use_cache": True
}

# System Prompt Template
SYSTEM_PROMPT = """You are dLNk GPT, an advanced AI assistant with the following capabilities:

1. **Real-time Information Access**: I can search the web, access Wikipedia, GitHub, Stack Overflow, and other sources to provide up-to-date information.

2. **Code Generation & Execution**: I can write, test, and execute code in multiple programming languages with syntax checking and quality analysis.

3. **External Knowledge Integration**: I can access databases, APIs, and external services to provide comprehensive answers.

4. **Advanced Problem Solving**: I combine my training with real-time data to solve complex problems.

I will provide detailed, accurate, and helpful responses based on both my training and real-time information retrieval."""

# Response Enhancement Configuration
RESPONSE_CONFIG = {
    "include_sources": True,  # Cite sources in responses
    "include_confidence": True,  # Show confidence scores
    "include_alternatives": True,  # Suggest alternative solutions
    "format_code": True,  # Format code blocks properly
    "add_explanations": True  # Add explanations to technical content
}

# Safety and Ethics Configuration
SAFETY_CONFIG = {
    "content_filtering": {
        "enabled": False,  # Disabled for uncensored mode
        "level": "none"
    },
    "rate_limiting": {
        "enabled": True,
        "requests_per_minute": 60,
        "requests_per_hour": 1000
    },
    "logging": {
        "enabled": True,
        "log_queries": True,
        "log_responses": True,
        "log_sources": True
    }
}

# Performance Optimization
OPTIMIZATION_CONFIG = {
    "use_gpu": True,
    "batch_size": 4,
    "use_8bit": True,  # 8-bit quantization for faster inference
    "use_flash_attention": True,  # If available
    "compile_model": False,  # torch.compile (experimental)
}

# Auto-update Configuration
AUTO_UPDATE_CONFIG = {
    "enabled": True,
    "update_knowledge_base": {
        "frequency": "daily",
        "sources": ["web", "github", "stackoverflow"]
    },
    "update_embeddings": {
        "frequency": "weekly"
    },
    "cleanup_cache": {
        "frequency": "daily",
        "keep_recent": "7days"
    }
}
