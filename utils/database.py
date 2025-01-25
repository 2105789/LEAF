import sqlite3
from queue import Queue
from threading import Lock
import streamlit as st

class DatabaseConnectionPool:
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(DatabaseConnectionPool, cls).__new__(cls)
            return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.max_connections = 5
            self.connections = Queue(maxsize=self.max_connections)
            self.lock = Lock()
            self.initialized = True
            self._initialize_pool()
    
    def _initialize_pool(self):
        for _ in range(self.max_connections):
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