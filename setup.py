from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'A package that allows to build neural networks models using my balls.'

# Setting up
setup(
    name="LorisBallsBasedModel",
    version=VERSION,
    author="Loris Pilotto",
    author_email="loris.pilotto.pm@gmail.com",
    description=DESCRIPTION,
    packages=find_packages(),
    install_requires=['tensorflow'],
    keywords=['python', 'neural network', 'tensorflow', 'LorisBallsBasedModel', 'machine learning'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)