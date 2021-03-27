from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='emeof-alexhf',
    version='0.0.1',
    author='Alexandre Hippert-Ferrer',
    author_email='alexandre.hippert-ferrer@centralesupelec.fr',
    scripts=['bin/emeof_script'],
    license='LICENSE.txt',
    description='The EM-EOF package for missing data reconstruction',
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    install_requires=[
        "Django >= 1.1.1",
        "pytest",
    ],
)
