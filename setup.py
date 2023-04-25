import os
import shutil
from setuptools import setup

import numpy

__name__ = 'sampling_tools'

# clean previous build
for root, dirs, files in os.walk("./sampling_tools/", topdown=False):
    for name in dirs:
        if (name == "build"):
            shutil.rmtree(name)

from os import path

release_info = {}
infopath = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        __name__, 'info.py'))

with open(infopath) as open_file:
    exec(open_file.read(), release_info)

this_directory = path.abspath(path.dirname(__file__))

def read_requirements(file):
    with open(file) as f:
        return f.read().splitlines()

def read_file(file):
   with open(file) as f:
        return f.read()

required = read_requirements("requirements/requirements-core.txt")

include_dirs = [numpy.get_include(),]

extra_link_args=[]

setup(
    name=__name__,
    author=release_info['__author__'],
    author_email=release_info['__email__'],
    version=release_info['__version__'],
    url=release_info['__url__'],
    packages=['sampling_tools'],
    install_requires=required,
    license=release_info['__license__'],
    description=release_info['__about__'],
    long_description=release_info['__long_description__'],
    classifiers=release_info['__classifiers__'],
    platforms=release_info['__platforms__']
)

