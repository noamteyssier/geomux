from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="geomux",
    version="0.2.4",
    author="Noam Teysser",
    author_email="Noam.Teyssier@ucsf.edu",
    packages=["geomux"],
    description="a tool to assign an identity to a table of barcode guides",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/noamteyssier/geomux",
    entry_points={"console_scripts": ["geomux = geomux.__main__:main_cli"]},
    install_requires=[
        "numpy",
        "pandas",
        "anndata",
        "scipy",
        "adjustpy",
    ],
)
