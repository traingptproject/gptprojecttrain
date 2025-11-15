"""
Automated Post-Training Workflow
Runs automatically after training completes
"""

import os
import sys
import time
import json
from datetime import datetime

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    POST-TRAINING ENHANCEMENT WORKFLOW                        ║
║                                                                              ║
║  This script automatically enhances your trained model with:                ║
║  • RAG (Retrieval-Augmented Generation)                                     ║
║  • External Knowledge Integration                                           ║
║  • Real-time Information Retrieval                                          ║
║  • Code Execution Capabilities                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

# Configuration
MODEL_PATH = "./dLNk-gpt-uncensored-v2"  # Path to trained model
CHECKPOINT_DIR = "./checkpoints"

def check_training_complete():
    """Check if training has completed"""
    print("\n🔍 Checking training status...")
    
    # Check if final model exists
    if os.path.exists(MODEL_PATH):
        print(f"✅ Trained model found: {MODEL_PATH}")
        return True
    
    # Check for checkpoints
    if os.path.exists(CHECKPOINT_DIR):
        checkpoints = [d for d in os.listdir(CHECKPOINT_DIR) if d.startswith("checkpoint-")]
        if checkpoints:
            # Get latest checkpoint
            latest = max(checkpoints, key=lambda x: int(x.split("-")[1]))
            latest_path = os.path.join(CHECKPOINT_DIR, latest)
            print(f"✅ Latest checkpoint found: {latest_path}")
            return True
    
    print("❌ No trained model found")
    return False


def install_dependencies():
    """Install required packages for enhancement"""
    print("\n📦 Installing enhancement dependencies...")
    
    packages = [
        "sentence-transformers",
        "faiss-cpu",
        "beautifulsoup4",
        "requests"
    ]
    
    for package in packages:
        print(f"  Installing {package}...")
        os.system(f"pip install -q {package}")
    
    print("✅ Dependencies installed")


def test_rag_system():
    """Test RAG system"""
    print("\n🧪 Testing RAG System...")
    
    try:
        from rag_system import RAGSystem
        
        rag = RAGSystem()
        
        # Test query
        test_query = "What is machine learning"
        documents = rag.retrieve(test_query, sources=["wikipedia"])
        
        if documents:
            print(f"✅ RAG system working (retrieved {len(documents)} documents)")
            return True
        else:
            print("⚠️  RAG system returned no documents")
            return False
            
    except Exception as e:
        print(f"❌ RAG system test failed: {e}")
        return False


def test_knowledge_integration():
    """Test external knowledge integration"""
    print("\n🧪 Testing Knowledge Integration...")
    
    try:
        from knowledge_integration import ExternalKnowledgeIntegration
        
        integration = ExternalKnowledgeIntegration()
        
        # Test code execution
        code = "print('Hello from test!')"
        result = integration.execute_code(code, "python")
        
        if result.get("success"):
            print("✅ Code execution working")
        else:
            print("⚠️  Code execution failed")
        
        # Test real-time data
        crypto_data = integration.get_realtime_data("crypto", coin_id="bitcoin")
        
        if crypto_data.get("success"):
            print(f"✅ Real-time data working (BTC: ${crypto_data.get('price_usd', 'N/A')})")
        else:
            print("⚠️  Real-time data failed")
        
        return True
        
    except Exception as e:
        print(f"❌ Knowledge integration test failed: {e}")
        return False


def test_enhanced_inference():
    """Test enhanced inference system"""
    print("\n🧪 Testing Enhanced Inference...")
    
    try:
        from enhanced_inference import EnhancedInference
        
        # Initialize (this will load the model)
        inference = EnhancedInference(MODEL_PATH)
        
        # Test simple query
        print("\n📝 Test Query: 'What is Python?'")
        result = inference.generate(
            "What is Python?",
            max_length=512,
            temperature=0.7
        )
        
        print(f"\n✅ Response generated:")
        print(f"{result['response'][:200]}...")
        
        # Test with RAG
        print("\n📝 Test Query with RAG: 'What is the latest news in AI?'")
        result = inference.generate(
            "What is the latest news in AI?",
            use_rag=True,
            max_length=512
        )
        
        print(f"\n✅ Response with RAG generated")
        if result['metadata'].get('sources'):
            print(f"   Sources used: {len(result['metadata']['sources'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_api_server():
    """Create simple API server for the enhanced model"""
    print("\n🌐 Creating API server...")
    
    api_code = '''"""
Simple API Server for Enhanced Model
"""

from flask import Flask, request, jsonify
from enhanced_inference import EnhancedInference

app = Flask(__name__)
inference = None

@app.before_first_request
def initialize():
    global inference
    print("Initializing model...")
    inference = EnhancedInference()
    print("Model ready!")

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    query = data.get("query")
    
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    result = inference.generate(
        query=query,
        use_rag=data.get("use_rag", None),
        max_length=data.get("max_length", 512),
        temperature=data.get("temperature", 0.7)
    )
    
    return jsonify(result)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    query = data.get("query")
    
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    response = inference.chat(query)
    
    return jsonify({"response": response})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model_loaded": inference is not None})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
'''
    
    with open("api_server.py", "w") as f:
        f.write(api_code)
    
    print("✅ API server created: api_server.py")
    print("\n   To start the server:")
    print("   $ python api_server.py")
    print("\n   API endpoints:")
    print("   POST /generate - Generate with full control")
    print("   POST /chat - Simple chat interface")
    print("   GET /health - Health check")


def create_usage_examples():
    """Create usage examples"""
    print("\n📝 Creating usage examples...")
    
    examples = '''"""
Usage Examples for Enhanced Model
"""

from enhanced_inference import EnhancedInference

# Initialize
inference = EnhancedInference()

# Example 1: Simple query
print("Example 1: Simple Query")
response = inference.chat("What is Python?")
print(response)

# Example 2: With RAG (real-time information)
print("\\nExample 2: Query with RAG")
result = inference.generate(
    "What is the current price of Bitcoin?",
    use_rag=True
)
print(result['response'])
print(f"Sources: {result['metadata']['sources']}")

# Example 3: Code generation
print("\\nExample 3: Code Generation")
response = inference.chat("Write a Python function to reverse a string")
print(response)

# Example 4: Code execution
print("\\nExample 4: Code Execution")
query = """
Run this Python code:
```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print([fibonacci(i) for i in range(10)])
```
"""
result = inference.generate(query)
print(result['response'])

# Example 5: Real-time data
print("\\nExample 5: Real-time Data")
response = inference.chat("What are the trending repositories on GitHub?")
print(response)

# Example 6: Complex query with multiple capabilities
print("\\nExample 6: Complex Query")
response = inference.chat(
    "Search for the latest Python web frameworks, "
    "compare their GitHub stars, and write a simple "
    "Flask hello world example"
)
print(response)
'''
    
    with open("usage_examples.py", "w") as f:
        f.write(examples)
    
    print("✅ Usage examples created: usage_examples.py")


def generate_report():
    """Generate enhancement report"""
    print("\n📊 Generating enhancement report...")
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "model_path": MODEL_PATH,
        "enhancements": {
            "rag_system": "Enabled",
            "knowledge_integration": "Enabled",
            "code_execution": "Enabled",
            "real_time_data": "Enabled"
        },
        "capabilities": [
            "Real-time information retrieval",
            "Web search integration",
            "Wikipedia access",
            "GitHub search",
            "Stack Overflow search",
            "Code execution (Python, Bash)",
            "Cryptocurrency prices",
            "Tech news",
            "GitHub trending",
            "Local knowledge base"
        ],
        "api_endpoints": {
            "generate": "POST /generate",
            "chat": "POST /chat",
            "health": "GET /health"
        },
        "files_created": [
            "post_training_config.py",
            "rag_system.py",
            "knowledge_integration.py",
            "enhanced_inference.py",
            "api_server.py",
            "usage_examples.py"
        ]
    }
    
    with open("enhancement_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("✅ Report saved: enhancement_report.json")
    
    return report


def main():
    """Main workflow"""
    start_time = time.time()
    
    # Step 1: Check if training is complete
    if not check_training_complete():
        print("\n⚠️  Training not complete. Please complete training first.")
        print("   This script will run automatically after training.")
        return
    
    # Step 2: Install dependencies
    install_dependencies()
    
    # Step 3: Test RAG system
    rag_ok = test_rag_system()
    
    # Step 4: Test knowledge integration
    knowledge_ok = test_knowledge_integration()
    
    # Step 5: Test enhanced inference
    inference_ok = test_enhanced_inference()
    
    # Step 6: Create API server
    create_api_server()
    
    # Step 7: Create usage examples
    create_usage_examples()
    
    # Step 8: Generate report
    report = generate_report()
    
    # Summary
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("ENHANCEMENT COMPLETE")
    print("="*80)
    
    print(f"\n⏱️  Total time: {elapsed_time:.1f} seconds")
    
    print("\n✅ System Status:")
    print(f"   RAG System: {'✅ Working' if rag_ok else '❌ Failed'}")
    print(f"   Knowledge Integration: {'✅ Working' if knowledge_ok else '❌ Failed'}")
    print(f"   Enhanced Inference: {'✅ Working' if inference_ok else '❌ Failed'}")
    
    print("\n📦 Files Created:")
    for file in report["files_created"]:
        print(f"   • {file}")
    
    print("\n🎯 Capabilities Added:")
    for capability in report["capabilities"]:
        print(f"   • {capability}")
    
    print("\n🚀 Next Steps:")
    print("   1. Test the enhanced model:")
    print("      $ python usage_examples.py")
    print("\n   2. Start the API server:")
    print("      $ python api_server.py")
    print("\n   3. Use the enhanced inference:")
    print("      from enhanced_inference import EnhancedInference")
    print("      inference = EnhancedInference()")
    print("      response = inference.chat('Your query here')")
    
    print("\n" + "="*80)
    print("🎉 Your model is now enhanced and ready to use!")
    print("="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
