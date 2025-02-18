from setuptools import setup, find_packages

setup(
    name='uhsr-retrieval',
    version='0.1.0',
    description='Unified Hyperbolic Spectral Retrieval (UHSR) - a novel text retrieval algorithm combining lexical and semantic search.',
    author='Vedaant Singh',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/uhsr-retrieval',
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
    ],
)
