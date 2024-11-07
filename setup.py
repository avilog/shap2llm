"""
shap2llm installation script.
"""

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read the version from the version file
version = {}
with open("t2ebm/version.py") as fp:
    exec(fp.read(), version)

setuptools.setup(
    name="shap2llm",
    version=version["__version__"],
    author="Avi Levin",
    author_email="avilog@gmail.com",
    description="A Natural Language Interface to model explanations produced with Shap",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "."},
    packages=["t2ebm"],
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.14.0,<2.0.0",
        "matplotlib",
        "shap",
        "openai>=1.52.0",
        "matplotlib"
    ],
)
