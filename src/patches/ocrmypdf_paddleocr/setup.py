from setuptools import setup, find_packages

setup(
    name="ocrmypdf-paddleocr",
    version="0.2.0.dev0",
    packages=find_packages(),
    install_requires=[
        "ocrmypdf",
        "paddleocr",
        "pillow",
    ],
    entry_points={
        "ocrmypdf": ["paddleocr = ocrmypdf_paddleocr.plugin"],
    },
)
