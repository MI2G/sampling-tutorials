# -*- coding: utf-8 -*-

""" PACKAGE INFO
This file contains parameters to fill settings in setup.py for 
the sampling-tools package.
"""

# Set the package release version
version_info = (0, 0, 1)
__version__ = '.'.join(str(c) for c in version_info)

# Set the package details
__name__                = 'sampling-tutorials'
__maintainer__          = "MI2G group: S. Melidonis, T. Klatzer, P. Dobson, C. Kemajou"
__maintainer_email__    = "t.klatzer@sms.ed.ac.uk"
__license__             = "GNU General Public License v3.0"
__platforms__           = "Linux, Windows, MacOS"
__provides__            = ["sampling_tools"]
__author__       = "MI2G group: S. Melidonis, T. Klatzer, P. Dobson, C. Kemajou"
__email__        = "t.klatzer@sms.ed.ac.uk"
__year__         = '2023'
__url__          = "https://github.com/MI2G/sampling-tutorials"
__download_url__ = "https://github.com/MI2G/sampling-tutorials"
__description__  = 'Tools for sampling tutorials'


# Default package properties
__about__ = ('{}\nAuthor: {} \nEmail: {} \nYear: {} \nInfo: {}'
             ''.format(__name__, __author__, __email__, __year__,
                       __description__))



__classifiers__ = ["Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: GNU General Public License v3.0",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Windows",      
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers"],


__long_description__ = """
==========
MI2G sampling-tutorials
==========
Imaging Inverse Problems and Bayesian Computation - Python tutorials to learn 
about (accelerated) sampling for uncertainty quantification and other advanced inferences.
Licensed under the terms of the GNU General Public License v3.0.
Information can be found on the Github main repository,
https://github.com/MI2G/sampling-tutorials
"""


