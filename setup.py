from setuptools import setup, find_packages

setup(
    name="asarrayflow",
    version="0.1.0",
    description="Minimal deep learning library using NumPy/CuPy",
    author="Aleksandr Gomelskii Kramar",
    packages=find_packages(where="."),
    package_dir={"": "."},
    install_requires=[
        "numpy>=1.21",
        "cupy>=11.0.0; platform_system != 'Windows'"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
)  
