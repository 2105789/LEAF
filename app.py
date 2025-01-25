import streamlit as st
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient
from langchain.prompts import PromptTemplate
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
import google.generativeai as genai
import asyncio
import nest_asyncio
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta
import re
import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential
import pandas as pd
import sqlite3
import concurrent.futures
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
from cachetools import TTLCache, LRUCache
from queue import Queue
from threading import Lock
import aiohttp
from functools import lru_cache
import time
import threading

nest_asyncio.apply()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize API keys from Streamlit secrets with backups
api_keys = {
    "GROQ_API_KEY": [st.secrets["GROQ_API_KEY"], st.secrets.get("GROQ_API_KEY_BACKUP", "")],
    "GEMINI_API_KEY": [st.secrets["GOOGLE_API_KEY"], st.secrets.get("GOOGLE_API_KEY_BACKUP", "")],
    "TAVILY_API_KEY": st.secrets["TAVILY_API_KEY"],
    "QDRANT_URL": st.secrets["QDRANT_URL"],
    "QDRANT_API_KEY": st.secrets["QDRANT_API_KEY"]
}

# Add connection pool for SQLite
class DatabaseConnectionPool:
    def __init__(self, max_connections=5):
        self.max_connections = max_connections
        self.connections = Queue(maxsize=max_connections)
        self.lock = Lock()
        
        # Initialize the pool
        for _ in range(max_connections):
            conn = sqlite3.connect("chat_history.db", check_same_thread=False)
            self.connections.put(conn)
    
    def get_connection(self):
        return self.connections.get()
    
    def return_connection(self, conn):
        self.connections.put(conn)
    
    def close_all(self):
        while not self.connections.empty():
            conn = self.connections.get()
            conn.close()

# Initialize the connection pool
db_pool = DatabaseConnectionPool()

def initialize_chat_history_db():
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            content TEXT NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS assistant_responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            content TEXT NOT NULL,
            user_message_id INTEGER,
            FOREIGN KEY(user_message_id) REFERENCES user_messages(id)
        )
    """)
    conn.commit()
    conn.close()

def save_user_message(content: str) -> int:
    conn = db_pool.get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO user_messages (content)
            VALUES (?)
        """, (content.lower(),))
        user_message_id = cursor.lastrowid
        conn.commit()
        return user_message_id
    finally:
        db_pool.return_connection(conn)

def save_assistant_response(content: str, user_message_id: int):
    conn = db_pool.get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO assistant_responses (content, user_message_id)
            VALUES (?, ?)
        """, (content, user_message_id))
        conn.commit()
    finally:
        db_pool.return_connection(conn)

def get_recent_response(user_message: str) -> Optional[str]:
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()
    one_week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute("""
        SELECT ar.content 
        FROM assistant_responses ar
        JOIN user_messages um ON ar.user_message_id = um.id
        WHERE um.content = ? AND um.timestamp >= ?
        ORDER BY um.timestamp DESC
        LIMIT 1
    """, (user_message.lower(), one_week_ago))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None

def load_chat_history():
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT um.content AS user_message, ar.content AS assistant_response
        FROM user_messages um
        LEFT JOIN assistant_responses ar ON um.id = ar.user_message_id
        ORDER BY um.timestamp ASC
    """)
    rows = cursor.fetchall()
    conn.close()
    return [{"role": "user", "content": row[0]} if row[0] else {"role": "assistant", "content": row[1]} for row in rows]

initialize_chat_history_db()

class MultiRAGChatbot:
    def __init__(self, model_config: Optional[Dict] = None):
        # Initialize state tracking variables first
        self.current_groq_key_index = 0
        self.current_gemini_key_index = 0
        self.key_switch_lock = Lock()
        
        # Initialize caches and other attributes
        self.embedding_cache = TTLCache(maxsize=1000, ttl=3600)  # 1 hour TTL
        self.search_cache = TTLCache(maxsize=100, ttl=300)  # 5 minutes TTL
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.session = None
        self.request_semaphore = asyncio.Semaphore(5)
        
        # Setup core components
        self.api_keys = api_keys
        self.setup_apis()
        self.setup_llms(model_config)
        self.setup_databases()
        self.load_prompts()
        self.initialize_response_cache()
        
        # Add rate limit tracking
        self.rate_limit_tracking = {
            "groq": {"last_error": None, "consecutive_errors": 0},
            "gemini": {"last_error": None, "consecutive_errors": 0}
        }

    def setup_apis(self):
        """Setup API configurations"""
        try:
            # Handle list-type API keys
            for key, value in self.api_keys.items():
                if isinstance(value, list):
                    os.environ[key] = value[0] if value and value[0] else ""
                else:
                    os.environ[key] = value
            
            if self.api_keys["GEMINI_API_KEY"][0]:
                genai.configure(api_key=self.api_keys["GEMINI_API_KEY"][0])
            
            if self.api_keys["TAVILY_API_KEY"]:
                self.tavily_client = TavilyClient(self.api_keys["TAVILY_API_KEY"])
            else:
                logger.warning("Tavily API key not found")
                
        except Exception as e:
            logger.error(f"Error in setup_apis: {str(e)}")
            raise

    def get_next_api_key(self, service: str) -> str:
        """Rotate to next available API key for the specified service"""
        with self.key_switch_lock:
            if service == "groq":
                self.current_groq_key_index = (self.current_groq_key_index + 1) % len(self.api_keys["GROQ_API_KEY"])
                return self.api_keys["GROQ_API_KEY"][self.current_groq_key_index]
            elif service == "gemini":
                self.current_gemini_key_index = (self.current_gemini_key_index + 1) % len(self.api_keys["GEMINI_API_KEY"])
                return self.api_keys["GEMINI_API_KEY"][self.current_gemini_key_index]

    def handle_rate_limit(self, service: str) -> bool:
        """
        Handle rate limit errors and rotate API keys
        Returns True if successfully rotated to new key, False if all keys exhausted
        """
        tracking = self.rate_limit_tracking[service]
        tracking["consecutive_errors"] += 1
        tracking["last_error"] = time.time()
        
        # Try switching to backup key
        if service == "groq":
            available_keys = [k for k in self.api_keys["GROQ_API_KEY"] if k]
            if self.current_groq_key_index + 1 < len(available_keys):
                self.current_groq_key_index += 1
                os.environ["GROQ_API_KEY"] = available_keys[self.current_groq_key_index]
                return True
        elif service == "gemini":
            available_keys = [k for k in self.api_keys["GEMINI_API_KEY"] if k]
            if self.current_gemini_key_index + 1 < len(available_keys):
                self.current_gemini_key_index += 1
                key = available_keys[self.current_gemini_key_index]
                os.environ["GEMINI_API_KEY"] = key
                genai.configure(api_key=key)
                return True
        return False

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def setup_llm_with_retry(self, service: str):
        """Setup LLM with enhanced retry logic and key rotation"""
        max_retries = len([k for k in self.api_keys[f"{'GEMINI' if service == 'gemini' else service.upper()}_API_KEY"] if k])
        last_error = None
        
        for attempt in range(max_retries):
            try:
                if service == "groq":
                    key = self.api_keys["GROQ_API_KEY"][self.current_groq_key_index]
                    if not key:
                        if not self.handle_rate_limit("groq"):
                            raise Exception("All Groq API keys exhausted")
                        continue
                    return ChatGroq(
                        model_name="mixtral-8x7b-32768",
                        groq_api_key=key,
                        temperature=0.7,
                        max_tokens=8192,
                        timeout=30,
                        max_retries=2,
                    )
                elif service == "gemini":
                    key = self.api_keys["GEMINI_API_KEY"][self.current_gemini_key_index]
                    if not key:
                        if not self.handle_rate_limit("gemini"):
                            raise Exception("All Gemini API keys exhausted")
                        continue
                    return ChatGoogleGenerativeAI(
                        model="gemini-1.0-pro",
                        google_api_key=key,
                        temperature=0.7,
                        top_p=0.95,
                        top_k=40,
                        max_output_tokens=8192,
                    )
            except Exception as e:
                last_error = e
                logger.error(f"Error setting up {service} LLM (attempt {attempt + 1}/{max_retries}): {str(e)}")
                
                # Check for rate limit errors
                if "429" in str(e) or "Resource" in str(e):
                    if not self.handle_rate_limit(service):
                        break  # No more keys available
                    continue
                    
                if attempt < max_retries - 1:
                    next_key = self.get_next_api_key(service)
                    if next_key:
                        logger.info(f"Switching to next {service} API key")
                        if service == "groq":
                            os.environ["GROQ_API_KEY"] = next_key
                        else:
                            os.environ["GEMINI_API_KEY"] = next_key
                            genai.configure(api_key=next_key)
                    
        raise last_error or Exception(f"Failed to initialize {service} LLM after all attempts")

    def setup_llms(self, model_config: Optional[Dict] = None):
        """Setup LLMs with error handling and fallbacks"""
        llm_setup_success = False
        
        # Try to set up Groq
        try:
            self.groq_llm = self.setup_llm_with_retry("groq")
            llm_setup_success = True
        except Exception as e:
            logger.error(f"Failed to setup Groq LLM: {str(e)}")
            self.groq_llm = None

        # Try to set up Gemini
        try:
            self.gemini_llm = self.setup_llm_with_retry("gemini")
            llm_setup_success = True
        except Exception as e:
            logger.error(f"Failed to setup Gemini LLM: {str(e)}")
            self.gemini_llm = None

        if not llm_setup_success:
            raise RuntimeError("Failed to initialize any language models")

        self.output_parser = StrOutputParser()

    async def handle_llm_error(self, service: str, error: Exception):
        """Handle LLM errors with retries and fallbacks"""
        logger.error(f"Error with {service}: {str(error)}")
        try:
            if service == "groq":
                self.groq_llm = self.setup_llm_with_retry("groq")
            else:
                self.gemini_llm = self.setup_llm_with_retry("gemini")
        except Exception as e:
            logger.error(f"Failed to recover {service} LLM: {str(e)}")
            return "I apologize, but I'm having trouble processing your request. Please try again in a moment."

    def setup_databases(self):
        """Setup vector database connection"""
        try:
            self.qdrant_client = QdrantClient(
                url=self.api_keys["QDRANT_URL"],
                api_key=self.api_keys["QDRANT_API_KEY"]
            )
            self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            self.vectorstore = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name="my_collection",
                embedding=self.embeddings
            )
            self._ensure_qdrant_collection()
        except Exception as e:
            logger.error(f"Error setting up databases: {str(e)}")
            raise

    def _ensure_qdrant_collection(self):
        try:
            self.qdrant_client.get_collection("my_collection")
        except Exception:
            self.qdrant_client.recreate_collection(
                collection_name="my_collection",
                vectors_config=models.VectorParams(
                    size=384,
                    distance=models.Distance.COSINE
                )
            )

    def load_prompts(self):
        self.prompts = {}
        prompt_files = {
            "system": "prompts/system_prompt.txt", 
            "routing": "prompts/routing_prompt.txt",
            "tavily": "prompts/tavily_response_prompt.txt",
            "qdrant": "prompts/qdrant_response_prompt.txt",
            "conversation": "prompts/conversation_prompt.txt",
            "ai_knowledge": "prompts/ai_knowledge_prompt.txt",
            "combined": "prompts/combined_response_prompt.txt" 
        }
        for key, filepath in prompt_files.items():
            try:
                with open(filepath, "r",encoding="utf-8") as f:
                    self.prompts[key] = f.read()
            except FileNotFoundError:
                if key == "combined":
                    self.prompts[key] = """
                    Given the following information from different sources, provide a comprehensive and coherent response to the query: {query}

                    Web Search Results:
                    {tavily_results}

                    Vector Database Results:
                    {qdrant_results}

                    AI Knowledge:
                    {ai_knowledge_results}

                    Please synthesize this information into a clear, accurate, and helpful response. Include relevant specific details from the sources while maintaining a natural conversational tone.
                    """
                else:
                    self.prompts[key] = None
        current_date = datetime.now().strftime("%A, %B %d, %Y")
        if self.prompts["system"]:
            self.prompts["system"] = self.prompts["system"].format(current_date=current_date)

    def initialize_response_cache(self):
        if "response_cache" not in st.session_state:
            st.session_state.response_cache = {}

    def get_flows(self, query: str) -> List[str]:
        if not self.prompts["routing"]:
            return ["Vector Database", "Web Search"]
        flow_prompt = PromptTemplate(
            template=self.prompts["routing"],
            input_variables=["query"]
        )
        flow_chain = flow_prompt | self.groq_llm | self.output_parser
        flows = flow_chain.invoke({"query": query})
        selected_flows = [flow.strip() for flow in flows.split(",")]
        if "AI Knowledge" in selected_flows:
            if len(selected_flows) == 1:
                selected_flows.append("Web Search")
            elif "None" in selected_flows:
                selected_flows.remove("None")
        if "Web Search" in selected_flows:
            if len(selected_flows) == 1 or (len(selected_flows) == 2 and "None" in selected_flows):
                selected_flows = ["Vector Database", "Web Search"]
            elif "None" in selected_flows:
                selected_flows.remove("None")
        if len(selected_flows) > 1 and "None" in selected_flows:
            selected_flows.remove("None")
        print(f"Selected flows for query '{query}': {selected_flows}")
        return selected_flows

    async def setup_aiohttp_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession()
            
    async def cleanup(self):
        if self.session:
            await self.session.close()
            
    @lru_cache(maxsize=100)
    def get_embedding(self, text: str):
        """Cache embeddings for frequently used texts"""
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        embedding = self.embeddings.embed_query(text)
        self.embedding_cache[text] = embedding
        return embedding

    async def _async_tavily_search(self, query: str):
        cache_key = f"tavily_{query}"
        if cache_key in self.search_cache:
            return self.search_cache[cache_key]

        async with self.request_semaphore:  # Limit concurrent requests
            try:
                search_params = {
                    "query": query,
                    "include_answer": True,
                    "include_raw_content": True,
                    "search_depth": "advanced",
                    "include_domains": [
                        "apctt.org", "arxiv.org", "springer.com", "nature.com",
                        "sciencemag.org", "ipcc.ch", "unfccc.int", "globalchange.gov",
                        "climate.gov", "carbonbrief.org", "wmo.int", 
                        "earthobservatory.nasa.gov", "copernicus.eu", "iea.org",
                        "irena.org", "pnas.org", "journals.ametsoc.org",
                        "sciencedirect.com", "tandfonline.com", "agu.org"
                    ],
                    "max_results": 15
                }
                
                result = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    partial(self.tavily_client.search, **search_params)
                )
                
                self.search_cache[cache_key] = result
                return result
                
            except Exception as e:
                logger.error(f"Tavily search error: {e}")
                return None

    async def execute_flows(self, query: str, flows: List[str]) -> Dict:
        results = {}
        cache_key = f"{query}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        async def process_web_search():
            if "Web Search" in flows:
                results["tavily"] = await self._async_tavily_search(query)
                
        async def process_vector_search():
            if "Vector Database" in flows:
                try:
                    embedding = self.get_embedding(query)
                    docs = await asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        lambda: self.vectorstore.similarity_search(query)
                    )
                    results["qdrant"] = "\n\n".join([doc.page_content for doc in docs]) if docs else ""
                except Exception as e:
                    logger.error(f"Vector search error: {e}")
                    results["qdrant"] = None
                    
        async def process_ai_knowledge():
            if "AI Knowledge" in flows:
                try:
                    results["ai_knowledge"] = await asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        lambda: self.generate_from_own_knowledge(query)
                    )
                except Exception as e:
                    logger.error(f"AI Knowledge error: {e}")
                    results["ai_knowledge"] = None

        # Execute all flows concurrently
        await asyncio.gather(
            process_web_search(),
            process_vector_search(),
            process_ai_knowledge()
        )

        st.session_state.response_cache[cache_key] = results
        return results

    def generate_from_own_knowledge(self, query: str) -> str:
        if not self.prompts["ai_knowledge"]:
            return "I'm sorry, I don't have enough information to answer that."
        context = self.prompts["system"] + "\n\n" + self.prompts["ai_knowledge"]
        ai_prompt = PromptTemplate(
            template=context,
            input_variables=["query"]
        )
        ai_chain = ai_prompt | self.groq_llm | self.output_parser
        return ai_chain.invoke({"query": query})

    async def combine_results(self, query: str, results: Dict, conversation_history: List[Dict[str, str]] = None) -> str:
        if not results:
            return "No results found from any knowledge source."
        
        try:
            # Try Gemini first
            response = await self._try_llm_with_fallback(
                "gemini", 
                query, 
                results, 
                conversation_history
            )
            return response
        except Exception as e:
            logger.error(f"Both LLMs failed: {str(e)}")
            return "I apologize, but I'm having technical difficulties. Please try again in a moment."

    async def _try_llm_with_fallback(self, primary_llm: str, query: str, results: Dict, conversation_history: List[Dict[str, str]]) -> str:
        """Try primary LLM with fallback to secondary"""
        try:
            formatted_history = self._format_conversation_history(conversation_history)
            context = self.prompts["system"] + "\n\n" + self.prompts["combined"]
            combine_prompt = PromptTemplate(
                template=context + "\n\nConversation History:\n{history}\n\nCurrent Query: {query}",
                input_variables=["query", "tavily_results", "qdrant_results", "ai_knowledge_results", "history"]
            )
            
            template_vars = {
                "query": query,
                "tavily_results": results.get("tavily", "No results from Tavily."),
                "qdrant_results": results.get("qdrant", "No results from Qdrant."),
                "ai_knowledge_results": results.get("ai_knowledge", "No results from AI Knowledge."),
                "history": formatted_history
            }
            
            llm = self.gemini_llm if primary_llm == "gemini" else self.groq_llm
            fallback_llm = self.groq_llm if primary_llm == "gemini" else self.gemini_llm
            
            try:
                refine_chain = combine_prompt | llm | self.output_parser
                return await refine_chain.ainvoke(template_vars)
            except Exception as e:
                if "429" in str(e) or "Resource" in str(e):
                    if self.handle_rate_limit(primary_llm):
                        # Retry with new key
                        llm = self.setup_llm_with_retry(primary_llm)
                        refine_chain = combine_prompt | llm | self.output_parser
                        return await refine_chain.ainvoke(template_vars)
                
                # Try fallback LLM
                logger.warning(f"Primary LLM failed, trying fallback: {str(e)}")
                refine_chain = combine_prompt | fallback_llm | self.output_parser
                return await refine_chain.ainvoke(template_vars)
                
        except Exception as e:
            raise Exception(f"Both LLMs failed: {str(e)}")

    def _format_conversation_history(self, conversation_history: List[Dict[str, str]]) -> str:
        if not conversation_history:
            return ""
        return "\n".join([f"{msg['role'].title()}: {msg['content']}" for msg in conversation_history[-5:]])

    def handle_conversation(self, query: str, conversation_history: List[Dict[str, str]] = None) -> str:
        try:
            if not self.prompts["conversation"]:
                return "I'm here to chat! How can I assist you today?"
            
            formatted_history = ""
            if conversation_history:
                for msg in conversation_history[-5:]:
                    formatted_history += f"{msg['role'].title()}: {msg['content']}\n"
            
            context = self.prompts["system"] + "\n\n" + self.prompts["conversation"]
            conversation_prompt = PromptTemplate(
                template=context + "\n\nConversation History:\n{history}\n\nCurrent Query: {query}",
                input_variables=["query", "history"]
            )
            
            # Try primary LLM first
            if self.groq_llm:
                try:
                    conversation_chain = conversation_prompt | self.groq_llm | self.output_parser
                    return conversation_chain.invoke({"query": query, "history": formatted_history})
                except Exception as e:
                    logger.warning(f"Primary LLM failed: {str(e)}")
            
            # Fallback to secondary LLM
            if self.gemini_llm:
                try:
                    conversation_chain = conversation_prompt | self.gemini_llm | self.output_parser
                    return conversation_chain.invoke({"query": query, "history": formatted_history})
                except Exception as e:
                    logger.error(f"Secondary LLM failed: {str(e)}")
            
            return "I apologize, but I'm having technical difficulties. Please try again in a moment."
                
        except Exception as e:
            logger.error(f"Conversation handling error: {str(e)}")
            return "I apologize, but I'm having technical difficulties. Please try asking your question again."

    def format_sources(self, results: Dict) -> str:
        sources = []
        if results.get("tavily"):
            for result in results["tavily"].get("results", []):
                sources.append(f"- [{result['title']}]({result['url']})")
        return "\n\n**Sources:**\n" + "\n".join(sources) if sources else ""

def process_agent_response(agent, prompt, response_queue):
    try:
        response = agent.get_response(prompt)
        response_queue.put((agent.name, response))
    except Exception as e:
        print(f"Error processing response for {agent.name}: {str(e)}")
        response_queue.put((agent.name, f"Error: {str(e)}"))

def get_combined_response(prompt, agents):
    response_queue = Queue()
    threads = []
    
    # Create threads for each agent
    for agent in agents:
        thread = threading.Thread(
            target=process_agent_response,
            args=(agent, prompt, response_queue)
        )
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Collect responses
    responses = {}
    while not response_queue.empty():
        agent_name, response = response_queue.get()
        responses[agent_name] = response
    
    # Format the combined response
    combined_response = ""
    for agent_name, response in responses.items():
        combined_response += f"\n\n{agent_name}'s Response:\n{response}"
    
    return combined_response

async def main():
    # Set page configuration
    st.set_page_config(
        page_title="LEAF Chatbot",
        page_icon="🌿",
        initial_sidebar_state="collapsed",
        layout="wide"
    )

    # Custom CSS for better styling
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width: 400px;
        margin-left: -400px;
    }
     
    """,unsafe_allow_html=True,)

    chatbot = MultiRAGChatbot()
    
    try:
        await chatbot.setup_aiohttp_session()

        if "messages" not in st.session_state:
            st.session_state.messages = []

        query_params = st.query_params
        user_id = str(hash(frozenset(query_params.items())))
        st.session_state.user_id = user_id

        st.title("🌿 LEAF Chatbot")
        st.markdown("Welcome to the LEAF Chatbot! Ask me anything about climate change, sustainability, and more.")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("What would you like to know?"):
            try:
                user_message_lower = prompt.lower()
                user_message_id = save_user_message(user_message_lower)
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    recent_response = get_recent_response(user_message_lower)
                    
                    if recent_response:
                        response = recent_response
                    else:
                        flows = chatbot.get_flows(prompt)
                        if "None" in flows:
                            response = chatbot.handle_conversation(prompt, st.session_state.messages)
                        else:
                            try:
                                results = await chatbot.execute_flows(prompt, flows)
                                response = await chatbot.combine_results(prompt, results, st.session_state.messages)
                                sources = chatbot.format_sources(results)
                                response = f"{response}\n\n{sources}"
                            except Exception as e:
                                logger.error(f"Error processing request: {str(e)}")
                                response = "I apologize, but I encountered an error. Please try rephrasing your question or try again later."
                    
                    save_assistant_response(response, user_message_id)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    message_placeholder.markdown(response)
                    
            except Exception as e:
                logger.error(f"Error in chat loop: {str(e)}")
                st.error("An error occurred while processing your message. Please try again.")

    except Exception as e:
        logger.error(f"Critical error in main: {str(e)}")
        st.error("The application encountered a critical error. Please refresh the page and try again.")
        
    finally:
        await chatbot.cleanup()
        db_pool.close_all()

if __name__ == "__main__":
    asyncio.run(main())
