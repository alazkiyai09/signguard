from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="signguard",
    version="0.1.0",
    author="Ahmad Whafa Azka Al Azkiyai",
    author_email="azka.alazkiyai@outlook.com",
    description="ECDSA-based cryptographic verification system for detecting poisoning attacks in federated learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alazkiyai09/signguard",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security :: Cryptography",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "signguard-server=signguard.server:main",
            "signguard-client=signguard.client:main",
            "signguard-simulate=signguard.simulate:main",
        ],
    },
)
