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

nest_asyncio.apply()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize API keys from Streamlit secrets
api_keys = {
    "GROQ_API_KEY": st.secrets["GROQ_API_KEY"],
    "GOOGLE_API_KEY": st.secrets["GOOGLE_API_KEY"],
    "TAVILY_API_KEY": st.secrets["TAVILY_API_KEY"],
    "QDRANT_URL": st.secrets["QDRANT_URL"],
    "QDRANT_API_KEY": st.secrets["QDRANT_API_KEY"]
}

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
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO user_messages (content)
        VALUES (?)
    """, (content.lower(),))
    user_message_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return user_message_id

def save_assistant_response(content: str, user_message_id: int):
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO assistant_responses (content, user_message_id)
        VALUES (?, ?)
    """, (content, user_message_id))
    conn.commit()
    conn.close()

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
        self.api_keys = api_keys
        self.setup_apis()
        self.setup_llms(model_config)
        self.setup_databases()
        self.load_prompts()
        self.initialize_response_cache()

    def setup_apis(self):
        for key, value in self.api_keys.items():
            os.environ[key] = value
        genai.configure(api_key=self.api_keys["GOOGLE_API_KEY"])
        self.tavily_client = TavilyClient(self.api_keys["TAVILY_API_KEY"])

    def setup_llms(self, model_config: Optional[Dict] = None):
        self.groq_llm = ChatGroq(
            model_name="mixtral-8x7b-32768",
            groq_api_key=self.api_keys["GROQ_API_KEY"],
            temperature=0.7,
            max_tokens=8192,
            timeout=None,
            max_retries=2,
        )
        self.gemini_llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=self.api_keys["GOOGLE_API_KEY"],
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            max_output_tokens=8192,
        )
        self.output_parser = StrOutputParser()

    def setup_databases(self):
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

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _async_tavily_search(self, query: str):
        search_params = {
            "query": query,
            "include_answer": True,
            "include_raw_content": True,
            "search_depth": "advanced",
            "include_domains": ["apctt.org", "arxiv.org", "springer.com", "nature.com", "sciencemag.org", "ipcc.ch", "unfccc.int", "globalchange.gov", "climate.gov", "carbonbrief.org", "wmo.int", "earthobservatory.nasa.gov", "copernicus.eu", "iea.org", "irena.org", "pnas.org", "journals.ametsoc.org", "sciencedirect.com", "tandfonline.com", "agu.org"],       
            "max_results": 15
        }
        return await asyncio.to_thread(self.tavily_client.search, **search_params)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def execute_flows(self, query: str, flows: List[str]) -> Dict:
        results = {}
        cache_key = f"{query}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"Executing flows for query '{query}': {flows}")
        if "Web Search" in flows:
            try:
                print("Executing Web Search flow...")
                tavily_results = await self._async_tavily_search(query)
                results["tavily"] = tavily_results
            except Exception:
                results["tavily"] = None
        if "Vector Database" in flows:
            try:
                print("Executing Vector Database flow...")
                docs = self.vectorstore.similarity_search(query)
                qdrant_results = "\n\n".join([doc.page_content for doc in docs]) if docs else ""
                results["qdrant"] = qdrant_results
            except Exception:
                results["qdrant"] = None
        if "AI Knowledge" in flows:
            try:
                print("Executing AI Knowledge flow...")
                ai_results = self.generate_from_own_knowledge(query)
                results["ai_knowledge"] = ai_results
            except Exception:
                results["ai_knowledge"] = None
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

    def combine_results(self, query: str, results: Dict, conversation_history: List[Dict[str, str]] = None) -> str:
        if not results:
            return "No results found from any knowledge source."
        
        # Format conversation history
        formatted_history = ""
        if conversation_history:
            for msg in conversation_history[-5:]:  # Only use last 5 messages for context
                formatted_history += f"{msg['role'].title()}: {msg['content']}\n"
        
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
        refine_chain = combine_prompt | self.gemini_llm | self.output_parser
        return refine_chain.invoke(template_vars)

    def handle_conversation(self, query: str, conversation_history: List[Dict[str, str]] = None) -> str:
        if not self.prompts["conversation"]:
            return "I'm here to chat! How can I assist you today?"
        
        # Format conversation history
        formatted_history = ""
        if conversation_history:
            for msg in conversation_history[-5:]:  # Only use last 5 messages for context
                formatted_history += f"{msg['role'].title()}: {msg['content']}\n"
        
        context = self.prompts["system"] + "\n\n" + self.prompts["conversation"]
        conversation_prompt = PromptTemplate(
            template=context + "\n\nConversation History:\n{history}\n\nCurrent Query: {query}",
            input_variables=["query", "history"]
        )
        conversation_chain = conversation_prompt | self.groq_llm | self.output_parser
        return conversation_chain.invoke({"query": query, "history": formatted_history})

    def format_sources(self, results: Dict) -> str:
        sources = []
        if results.get("tavily"):
            for result in results["tavily"].get("results", []):
                sources.append(f"- [{result['title']}]({result['url']})")
        return "\n\n**Sources:**\n" + "\n".join(sources) if sources else ""

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

    st.title("🌿 LEAF Chatbot")
    st.markdown("Welcome to the LEAF Chatbot! Ask me anything about climate change, sustainability, and more.")

    chatbot = MultiRAGChatbot()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    query_params = st.query_params
    user_id = str(hash(frozenset(query_params.items())))
    st.session_state.user_id = user_id

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What would you like to know?"):
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
                        response = chatbot.combine_results(prompt, results, st.session_state.messages)
                        sources = chatbot.format_sources(results)
                        response = f"{response}\n\n{sources}"
                    except Exception:
                        response = "Sorry, I encountered an error while processing your request. Please try again later."
            save_assistant_response(response, user_message_id)
            st.session_state.messages.append({"role": "assistant", "content": response})
            message_placeholder.markdown(response)

if __name__ == "__main__":
    asyncio.run(main())