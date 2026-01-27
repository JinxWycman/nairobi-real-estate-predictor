#!/bin/bash
echo "Starting Nairobi Real Estate Predictor..."

# Set default port if PORT is not set
PORT=
echo "Using PORT: "

# Run Streamlit with the port
streamlit run app/streamlit_app.py --server.port= --server.address=0.0.0.0
