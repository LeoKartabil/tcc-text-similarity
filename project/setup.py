from setuptools import setup, find_packages

setup(
    name="text-comparison-system",
    version="3.5.15",
    description="Sistema de Comparação de Conteúdo de Textos usando TF-IDF e SBERT",
    author="Leonardo Kartabil",
    author_email="leonardo.kar@hotmail.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "streamlit>=1.28.0",
        "pandas>=1.5.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "sentence-transformers>=2.2.0",
        "nltk>=3.8",
        "PyPDF2>=3.0.0",
        "pdfplumber>=0.9.0",
        "python-docx>=0.8.11",
        "matplotlib>=3.7.0",
        "plotly>=5.15.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ]
    },
    classifiers=[
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3.12.10",
    ],
)