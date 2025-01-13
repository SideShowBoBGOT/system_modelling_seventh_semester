from setuptools import setup, find_packages

setup(
    name='my_system_modelling_exam',
    version="0.1",
    python_requires=">=3.11",
    install_requires=[

    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)