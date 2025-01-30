import sqlite3
from queue import Queue
from threading import Lock
import streamlit as st

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

@st.cache_resource
def get_db_pool():
    return DatabaseConnectionPool()

db_pool = get_db_pool() 