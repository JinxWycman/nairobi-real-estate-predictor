from setuptools import setup, find_packages

setup(
    name="nairobi-real-estate-predictor",
    version="1.0.0",
    description="Nairobi Real Estate Price Predictor - ML Web App",
    author="Joseph Thuo",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.28.1",
        "pandas>=2.1.4",
        "numpy>=1.24.3",
        "matplotlib>=3.7.2",
        "seaborn>=0.12.2",
        "plotly>=5.17.0",
        "folium>=0.15.1",
        "geopy>=2.4.0",
        "scikit-learn>=1.3.2",
        "xgboost>=2.0.3",
        "lightgbm>=4.3.0"
    ],
    python_requires=">=3.9",
)
