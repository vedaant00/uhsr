from setuptools import setup, find_packages
import os

with open("PROJECT_DESCRIPTION.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='uhsr',
    version='0.1.0',
    description='Unified Hyperbolic Spectral Retrieval (UHSR) - a novel text retrieval algorithm combining lexical and semantic search.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Vedaant Singh',
    url='https://github.com/vedaant2000/uhsr-retrieval',
    keywords=[
        'uhsr', 'text retrieval', 'BM25', 'FAISS', 'semantic search', 
        'lexical search', 'spectral re-ranking', 'machine learning', 'NLP'
    ],
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'sentence-transformers'
    ],
    extras_require={
        'gpu': ['faiss-gpu'],
        'cpu': ['faiss-cpu']
    },
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
)
