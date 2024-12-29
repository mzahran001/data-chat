"""Configuration settings for the data chat application."""

import streamlit as st

# MongoDB Configuration
MONGODB_URI = f"mongodb+srv://{st.secrets['db_username']}:{st.secrets['db_password']}@cluster0.nzc35.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
DB_NAME = "purchase_orders"
COLLECTION_NAME = "order_data"

# Data Processing Configuration
BATCH_SIZE = 5000
CSV_FILE_PATH = "PURCHASE_ORDER_DATA_EXTRACT_2012-2015_0.csv"

# Application Settings
CACHE_TTL = 3600  # Cache lifetime in seconds
