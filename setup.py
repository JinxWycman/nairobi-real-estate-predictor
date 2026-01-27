from setuptools import setup, find_packages

setup(
    name='nairobi-real-estate',
    version='1.0',
    description='Nairobi Real Estate Price Predictor',
    author='Joseph Thuo',
    packages=find_packages(),
    install_requires=[
        'streamlit>=1.28',
        'pandas>=1.5',
        'numpy>=1.24',
        'scikit-learn>=1.3',
    ],
    python_requires='>=3.9',
)
