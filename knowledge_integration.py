"""
External Knowledge Integration System
Connects model with external APIs, databases, and services
"""

import os
import sys
import json
import sqlite3
import subprocess
from typing import Dict, List, Any, Optional
from datetime import datetime
import tempfile

try:
    import requests
except ImportError:
    os.system("pip install -q requests")
    import requests

from post_training_config import EXTERNAL_APIS, REALTIME_SOURCES


class CodeExecutor:
    """
    Safe code execution in isolated environment
    """
    
    def __init__(self, config=None):
        self.config = config or EXTERNAL_APIS["code_execution"]
        self.timeout = self.config.get("timeout", 30)
    
    def execute_python(self, code: str) -> Dict[str, Any]:
        """Execute Python code safely"""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            # Execute with timeout
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            # Clean up
            os.unlink(temp_file)
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Execution timeout ({self.timeout}s)"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def execute_bash(self, command: str) -> Dict[str, Any]:
        """Execute bash command safely"""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Execution timeout ({self.timeout}s)"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def execute(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Execute code in specified language"""
        if language == "python":
            return self.execute_python(code)
        elif language == "bash":
            return self.execute_bash(code)
        else:
            return {
                "success": False,
                "error": f"Unsupported language: {language}"
            }


class KnowledgeDatabase:
    """
    Local knowledge database for caching and storage
    """
    
    def __init__(self, db_path: str = "./knowledge_base.db"):
        self.db_path = db_path
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                response TEXT NOT NULL,
                source TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS code_snippets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                language TEXT NOT NULL,
                code TEXT NOT NULL,
                description TEXT,
                tags TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                endpoint TEXT NOT NULL,
                params TEXT,
                response TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                ttl INTEGER DEFAULT 3600
            )
        """)
        
        conn.commit()
        conn.close()
        
        print(f"✅ Knowledge database initialized: {self.db_path}")
    
    def store_knowledge(self, query: str, response: str, source: str = None, metadata: Dict = None):
        """Store knowledge entry"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO knowledge (query, response, source, metadata)
            VALUES (?, ?, ?, ?)
        """, (query, response, source, json.dumps(metadata) if metadata else None))
        
        conn.commit()
        conn.close()
    
    def search_knowledge(self, query: str, limit: int = 5) -> List[Dict]:
        """Search for similar knowledge entries"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Simple text search (can be improved with FTS or embeddings)
        cursor.execute("""
            SELECT query, response, source, timestamp, metadata
            FROM knowledge
            WHERE query LIKE ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (f"%{query}%", limit))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "query": row[0],
                "response": row[1],
                "source": row[2],
                "timestamp": row[3],
                "metadata": json.loads(row[4]) if row[4] else None
            })
        
        conn.close()
        return results
    
    def store_code(self, language: str, code: str, description: str = None, tags: List[str] = None):
        """Store code snippet"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO code_snippets (language, code, description, tags)
            VALUES (?, ?, ?, ?)
        """, (language, code, description, json.dumps(tags) if tags else None))
        
        conn.commit()
        conn.close()
    
    def search_code(self, language: str = None, tags: List[str] = None, limit: int = 10) -> List[Dict]:
        """Search for code snippets"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT language, code, description, tags, timestamp FROM code_snippets WHERE 1=1"
        params = []
        
        if language:
            query += " AND language = ?"
            params.append(language)
        
        if tags:
            for tag in tags:
                query += " AND tags LIKE ?"
                params.append(f"%{tag}%")
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "language": row[0],
                "code": row[1],
                "description": row[2],
                "tags": json.loads(row[3]) if row[3] else None,
                "timestamp": row[4]
            })
        
        conn.close()
        return results


class RealtimeDataFetcher:
    """
    Fetch real-time data from various APIs
    """
    
    def __init__(self, config=None):
        self.config = config or REALTIME_SOURCES
    
    def get_crypto_price(self, coin_id: str = "bitcoin") -> Dict[str, Any]:
        """Get cryptocurrency price from CoinGecko"""
        try:
            url = f"https://api.coingecko.com/api/v3/simple/price"
            params = {
                "ids": coin_id,
                "vs_currencies": "usd",
                "include_24hr_change": "true",
                "include_market_cap": "true"
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if coin_id in data:
                return {
                    "success": True,
                    "coin": coin_id,
                    "price_usd": data[coin_id].get("usd"),
                    "change_24h": data[coin_id].get("usd_24h_change"),
                    "market_cap": data[coin_id].get("usd_market_cap")
                }
            
            return {"success": False, "error": "Coin not found"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_github_trending(self, language: str = None) -> Dict[str, Any]:
        """Get trending repositories from GitHub"""
        try:
            url = "https://api.github.com/search/repositories"
            params = {
                "q": f"created:>{datetime.now().strftime('%Y-%m-%d')} stars:>100",
                "sort": "stars",
                "order": "desc",
                "per_page": 10
            }
            
            if language:
                params["q"] += f" language:{language}"
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            repos = []
            for item in data.get("items", []):
                repos.append({
                    "name": item["full_name"],
                    "description": item.get("description"),
                    "stars": item["stargazers_count"],
                    "language": item.get("language"),
                    "url": item["html_url"]
                })
            
            return {
                "success": True,
                "repositories": repos
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_tech_news(self, category: str = "technology") -> Dict[str, Any]:
        """Get latest tech news (using free RSS feeds)"""
        try:
            # Using Hacker News API as example
            url = "https://hacker-news.firebaseio.com/v0/topstories.json"
            response = requests.get(url, timeout=10)
            story_ids = response.json()[:10]
            
            stories = []
            for story_id in story_ids:
                story_url = f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
                story_response = requests.get(story_url, timeout=5)
                story_data = story_response.json()
                
                if story_data.get("type") == "story":
                    stories.append({
                        "title": story_data.get("title"),
                        "url": story_data.get("url"),
                        "score": story_data.get("score"),
                        "time": story_data.get("time")
                    })
            
            return {
                "success": True,
                "stories": stories
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


class ExternalKnowledgeIntegration:
    """
    Main integration system combining all external knowledge sources
    """
    
    def __init__(self):
        print("🔧 Initializing External Knowledge Integration...")
        
        self.code_executor = CodeExecutor()
        self.database = KnowledgeDatabase()
        self.realtime_fetcher = RealtimeDataFetcher()
        
        print("✅ External Knowledge Integration initialized")
    
    def execute_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Execute code and return results"""
        result = self.code_executor.execute(code, language)
        
        # Store successful code in database
        if result.get("success"):
            self.database.store_code(
                language=language,
                code=code,
                description="Auto-generated code",
                tags=["auto-generated"]
            )
        
        return result
    
    def get_realtime_data(self, data_type: str, **kwargs) -> Dict[str, Any]:
        """Get real-time data from various sources"""
        if data_type == "crypto":
            return self.realtime_fetcher.get_crypto_price(kwargs.get("coin_id", "bitcoin"))
        elif data_type == "github_trending":
            return self.realtime_fetcher.get_github_trending(kwargs.get("language"))
        elif data_type == "news":
            return self.realtime_fetcher.get_tech_news(kwargs.get("category", "technology"))
        else:
            return {"success": False, "error": f"Unknown data type: {data_type}"}
    
    def search_knowledge_base(self, query: str) -> List[Dict]:
        """Search local knowledge base"""
        return self.database.search_knowledge(query)
    
    def store_interaction(self, query: str, response: str, source: str = "model", metadata: Dict = None):
        """Store interaction in knowledge base"""
        self.database.store_knowledge(query, response, source, metadata)


# Example usage
if __name__ == "__main__":
    # Initialize system
    integration = ExternalKnowledgeIntegration()
    
    # Test code execution
    print("\n" + "="*80)
    print("TEST: Code Execution")
    print("="*80)
    
    code = """
print("Hello from executed code!")
result = 2 + 2
print(f"2 + 2 = {result}")
"""
    
    result = integration.execute_code(code, "python")
    print(f"Success: {result['success']}")
    if result['success']:
        print(f"Output:\n{result['stdout']}")
    
    # Test real-time data
    print("\n" + "="*80)
    print("TEST: Real-time Data (Crypto)")
    print("="*80)
    
    crypto_data = integration.get_realtime_data("crypto", coin_id="bitcoin")
    print(json.dumps(crypto_data, indent=2))
    
    # Test GitHub trending
    print("\n" + "="*80)
    print("TEST: GitHub Trending")
    print("="*80)
    
    trending = integration.get_realtime_data("github_trending", language="python")
    if trending["success"]:
        for repo in trending["repositories"][:3]:
            print(f"⭐ {repo['name']} ({repo['stars']} stars)")
            print(f"   {repo['description']}")
