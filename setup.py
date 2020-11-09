#! /usr/bin/env python
#

DESCRIPTION = "pyprosp: SED fitting using 'prospector' package, including a few posterior processes"
LONG_DESCRIPTION = "pyprosp: SED fitting using 'prospector' package, including a few posterior processes"

DISTNAME = 'pyprosp'
AUTHOR = 'Martin Briday'
MAINTAINER = 'Martin Briday' 
MAINTAINER_EMAIL = 'briday@ipnl.in2p3.fr'
URL = 'https://github.com/MartinBriday/pyprosp'
LICENSE = 'Apache 2.0'
DOWNLOAD_URL = 'https://github.com/MartinBriday/pyprosp'
VERSION = '1.0.4'

try:
    from setuptools import setup, find_packages
    _has_setuptools = True
except ImportError:
    from distutils.core import setup

def check_dependencies():
    install_requires = []

    try:
        import prospect
    except ImportError:
        install_requires.append('prospect')
       
     
    return install_requires

if __name__ == "__main__":

    install_requires = check_dependencies()

    if _has_setuptools:
        packages = find_packages()
        print(packages)
    else:
        # This should be updated if new submodules are added
        packages = ['pyprosp']

    setup(name=DISTNAME,
          author=AUTHOR,
          author_email=MAINTAINER_EMAIL,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          long_description=LONG_DESCRIPTION,
          license=LICENSE,
          url=URL,
          version=VERSION,
          download_url=DOWNLOAD_URL,
          install_requires=install_requires,
          packages=packages,
          package_data={},
          classifiers=[
              'Intended Audience :: Science/Research',
              'Programming Language :: Python :: 2.7',
              'Programming Language :: Python :: 3.5',              
              'License :: OSI Approved :: BSD License',
              'Topic :: Scientific/Engineering :: Astronomy',
              'Operating System :: POSIX',
              'Operating System :: Unix',
              'Operating System :: MacOS'],
      )










