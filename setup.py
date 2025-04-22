from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="neurashield",
    version="0.1.0",
    author="NeuraShield Team",
    author_email="team@neurashield.ai",
    description="Advanced network threat detection system with blockchain audit trail",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/neurashield",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Information Technology",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Security",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.5",
            "pytest-cov>=2.12.1",
            "black>=21.9b0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ],
    },
    entry_points={
        "console_scripts": [
            "neurashield-api=backend.src.api:main",
            "neurashield-dashboard=frontend.src.dashboard:main",
            "neurashield-processor=backend.src.traffic_processor:main",
        ],
    },
) 