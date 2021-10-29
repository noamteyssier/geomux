from setuptools import setup

setup(
    name="geomux",
    version="0.1.0",
    author="Noam Teysser",
    author_email="Noam.Teyssier@ucsf.edu",
    packages=["geomux"],
    description="a tool to assign an identity to a table of barcode guides",
    entry_points={'console_scripts': ['geomux = geomux.__main__:main_cli']},
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "plotly",
        "umap-learn"]
)
