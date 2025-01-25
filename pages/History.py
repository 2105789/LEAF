import streamlit as st

# Must be the first Streamlit command
st.set_page_config(
    page_title="LEaF Chatbot - History",
    page_icon="📊",
    layout="wide"
)

import plotly.express as px
import pandas as pd
from datetime import datetime
from utils.database import db_pool
import math

def execute_query(query, params=None):
    """Execute a database query in a thread-safe way"""
    conn = db_pool.get_connection()
    try:
        cursor = conn.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        return cursor.fetchall()
    finally:
        db_pool.return_connection(conn)

def paginate_messages(messages, page, per_page=10):
    """Helper function to paginate messages"""
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    return messages[start_idx:end_idx]

def show_pagination(total_items, current_page, per_page=10, key_prefix=""):
    """Display pagination controls"""
    total_pages = math.ceil(total_items / per_page)
    
    if total_pages <= 1:
        return current_page
    
    col1, col2, col3, col4, col5 = st.columns([2, 1, 2, 1, 2])
    
    with col1:
        if st.button("◀◀ First", key=f"{key_prefix}_first", disabled=current_page == 1):
            return 1
    with col2:
        if st.button("◀ Prev", key=f"{key_prefix}_prev", disabled=current_page == 1):
            return max(1, current_page - 1)
    with col3:
        st.write(f"Page {current_page} of {total_pages}")
    with col4:
        if st.button("Next ▶", key=f"{key_prefix}_next", disabled=current_page == total_pages):
            return min(total_pages, current_page + 1)
    with col5:
        if st.button("Last ▶▶", key=f"{key_prefix}_last", disabled=current_page == total_pages):
            return total_pages
            
    return current_page

def show_history_dashboard():
    st.title("💬 Chat History Dashboard")
    
    # Initialize session state for pagination
    if 'latest_page' not in st.session_state:
        st.session_state.latest_page = 1
    if 'faq_page' not in st.session_state:
        st.session_state.faq_page = 1
    
    # Create tabs
    tab1, tab2 = st.tabs(["Latest Messages", "Most Asked Questions"])
    
    with tab1:
        st.header("Recent Conversations")
        
        # Get total count first
        total_messages = execute_query("SELECT COUNT(*) FROM user_messages")[0][0]
        
        # Get paginated conversations
        latest_messages = execute_query("""
            SELECT 
                um.id as chat_id,
                um.timestamp as user_timestamp,
                um.content as user_message,
                ar.content as assistant_response
            FROM user_messages um
            LEFT JOIN assistant_responses ar ON um.id = ar.user_message_id
            ORDER BY um.timestamp DESC
            LIMIT ? OFFSET ?
        """, (10, (st.session_state.latest_page - 1) * 10))
        
        if latest_messages:
            for msg in latest_messages:
                timestamp = datetime.strptime(msg[1], '%Y-%m-%d %H:%M:%S')
                formatted_time = timestamp.strftime('%B %d, %Y %I:%M %p')
                
                preview = msg[2][:100] + "..." if len(msg[2]) > 100 else msg[2]
                with st.expander(f"🗣️ {preview} - {formatted_time}"):
                    st.markdown("**User Message:**")
                    st.write(msg[2])
                    st.markdown("**Assistant Response:**")
                    st.write(msg[3])
                    st.markdown(f"*Chat ID: {msg[0]}*", help="Reference ID for this conversation")
            
            # Show pagination controls
            st.session_state.latest_page = show_pagination(
                total_messages, 
                st.session_state.latest_page,
                key_prefix="latest"
            )
        else:
            st.info("No chat history available")
    
    with tab2:
        st.header("Most Asked Questions")
        
        # Get total count of frequent questions
        total_frequent = execute_query("""
            SELECT COUNT(*) FROM (
                SELECT COUNT(*) 
                FROM user_messages 
                WHERE timestamp >= datetime('now', '-30 days')
                GROUP BY LOWER(content)
                HAVING COUNT(*) > 1
            ) subquery
        """)[0][0]
        
        # Get paginated frequent questions
        frequent_questions = execute_query("""
            SELECT 
                content as question,
                COUNT(*) as frequency,
                GROUP_CONCAT(id) as chat_ids
            FROM user_messages
            WHERE timestamp >= datetime('now', '-30 days')
            GROUP BY LOWER(content)
            HAVING COUNT(*) > 1
            ORDER BY frequency DESC
            LIMIT ? OFFSET ?
        """, (10, (st.session_state.faq_page - 1) * 10))
        
        if frequent_questions:
            # Add statistics
            stats = execute_query("""
                SELECT 
                    COUNT(DISTINCT id) as total_chats,
                    COUNT(DISTINCT DATE(timestamp)) as active_days,
                    COUNT(DISTINCT content) as unique_questions
                FROM user_messages
                WHERE timestamp >= datetime('now', '-30 days')
            """)
            
            st.markdown("### 30-Day Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Conversations", stats[0][0])
            with col2:
                st.metric("Active Days", stats[0][1])
            with col3:
                st.metric("Unique Questions", stats[0][2])
            
            st.markdown("### Frequently Asked Questions")
            for question, freq, chat_ids in frequent_questions:
                with st.expander(f"❓ Asked {freq} times: {question[:100]}{'...' if len(question) > 100 else ''}"):
                    st.markdown("**Full Question:**")
                    st.write(question)
                    
                    recent_response = execute_query("""
                        SELECT ar.content, um.timestamp
                        FROM assistant_responses ar
                        JOIN user_messages um ON ar.user_message_id = um.id
                        WHERE LOWER(um.content) = LOWER(?)
                        ORDER BY um.timestamp DESC
                        LIMIT 1
                    """, (question,))
                    
                    if recent_response:
                        st.markdown("**Most Recent Response:**")
                        st.write(recent_response[0][0])
                        st.markdown(f"*Last asked: {recent_response[0][1]}*")
            
            # Show pagination controls
            st.session_state.faq_page = show_pagination(
                total_frequent, 
                st.session_state.faq_page,
                key_prefix="faq"
            )
        else:
            st.info("No frequently asked questions found in the last 30 days")

if __name__ == "__main__":
    show_history_dashboard() 