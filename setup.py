# from setuptools import setup, find_packages
# import numpy
#
# setup(
#     name="bayesbreak",
#     version="0.0.1",
#     author="Omid Shams Solari",
#     author_email="solari@berkeley.edu",
#     description="Bayesian Process Segmentation",
#     maintainer="Omid Shams Solari",
#     maintainer_email="solari@berkeley.edu",
#     setup_requires=[
#         # Setuptools 18.0 properly handles Cython extensions.
#         "setuptools>=18.0",
#         "cython",
#     ],
#     tests_require=["pytest"],
#     packages=find_packages(),
#     include_package_data=True,
#     include_dirs=[numpy.get_include()],
# )


import os
from setuptools import setup, find_packages
import numpy

# Read requirements from the existing pinned requirements file
requirements_path = os.path.join('etc', 'requirements', 'requirements.txt')
with open(requirements_path) as f:
    # Filter out comments and empty lines
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    # Remove any unsafe packages or version specifiers that setuptools can't handle
    requirements = [req for req in requirements if not req.startswith('# The following packages')]

setup(
    name="bayesbreak",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements,
    author="Omid Shams Solari",
    author_email="solari@berkeley.edu",
    description="Bayesian Process Segmentation",
    keywords="bayesian regression, change-point estimation",
    tests_require=["pytest"],
    include_package_data=True,
    include_dirs=[numpy.get_include()],
)

