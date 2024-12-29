# Data-Chat: Procurement Data Analysis Assistant

An interactive Streamlit-based application that allows users to analyze California Department of General Services (DGS) procurement data using natural language queries. The application combines the power of LangChain, PandasAI, and MongoDB to provide intelligent insights into procurement records from fiscal years 2012 to 2015.

## Features

- Natural language querying of procurement data
- Interactive data visualization capabilities
- MongoDB integration for efficient data storage
- Streamlit-based user interface
- Support for complex procurement data analysis

## Technology Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **Database**: MongoDB Atlas
- **AI/ML**: LangChain, PandasAI
- **Data Processing**: Pandas, Matplotlib

## Prerequisites

- Python 3.10+
- MongoDB Atlas account
- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/mzahran001/data-chat.git
cd data-chat
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
   - Set up MongoDB credentials in Streamlit secrets
   - Configure OpenAI API key

## Project Structure

```
data-chat/
├── src/
│   ├── chat.py          # Main application logic
│   ├── config.py        # Configuration settings
│   ├── mongodb.py       # MongoDB connection handler
│   └── eval.py          # Evaluation utilities
├── data/                # Data storage
├── .streamlit/          # Streamlit configuration
└── requirements.txt     # Project dependencies
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run src/chat.py
```

2. Access the application through your web browser at `http://localhost:8501`

3. Enter natural language queries to analyze procurement data

## Data Overview

The application processes procurement records with the following key information:
- Temporal coverage: Fiscal Years 2012-2015
- Supplier information and qualifications
- UNSPSC item classifications
- Acquisition methods and types
- Department details
- Price and quantity data