#!/bin/bash
echo "Starting Nairobi Real Estate Predictor..."
echo "PORT: "
echo "Python version: Python 3.14.0"
streamlit run app/streamlit_app.py --server.port= --server.address=0.0.0.0
