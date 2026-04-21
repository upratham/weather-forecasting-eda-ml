from setuptools import setup, find_packages

setup(
    name="global-weather-forecasting",
    version="0.1.0",
    description="End-to-end weather trend forecasting using EDA, anomaly detection, time series & ensemble ML models.",
    author="Your Name",
    author_email="your.email@example.com",
    python_requires=">=3.9",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "scikit-learn>=1.3.0",
        "statsmodels>=0.14.0",
        "xgboost>=1.7.0",
        "plotly>=5.15.0",
        "kaggle>=1.5.0",
        "joblib>=1.3.0",
        "gradio>=5.0.0",
        "prophet>=1.1.0",
    ],
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "notebook>=7.0.0",
            "ipykernel>=6.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
