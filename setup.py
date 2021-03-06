from setuptools import setup

setup(
    name="geomux",
    version="0.2.1",
    author="Noam Teysser",
    author_email="Noam.Teyssier@ucsf.edu",
    packages=["geomux"],
    description="a tool to assign an identity to a table of barcode guides",
    entry_points={'console_scripts': ['geomux = geomux.__main__:main_cli']},
    install_requires=[
        "numpy",
        "pandas",
        "anndata",
        "scipy",
        "plotly",
        "umap-learn"]
)
