from setuptools import find_packages, setup


setup(
    name="cfaf-ddct-reproducibility",
    version="0.1.0",
    packages=find_packages("."),
    py_modules=["main", "synthetic_data"],
    python_requires=">=3.8",
    install_requires=["numpy>=1.21", "torch>=1.12"],
)
