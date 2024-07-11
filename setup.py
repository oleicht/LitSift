from pathlib import Path
from setuptools import setup, find_packages

def parse_requirements(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="LitSift",
    version="0.1.0",
    description="Simple UI to find relevant papers ",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author="Oliver Leicht",
    author_email="oliverleicht@gmx.de",
    url="https://github.com/oleicht/LitSift",
    packages=find_packages(),
    install_requires=parse_requirements(Path(__file__).parent / "requirements.txt"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'lit-sift=app.main:main',
        ],
    },
)
