from setuptools import setup, find_packages

setup(
    name="iclr_agent",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'torch>=1.9.0',
        'matplotlib>=3.4.3',
        'networkx>=2.6.3',
        'scikit-learn>=0.24.2',
        'pandas>=1.3.0',
        'seaborn>=0.11.2',
        'tqdm>=4.62.3'
    ]
)
