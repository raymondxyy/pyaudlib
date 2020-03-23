from __future__ import print_function

import os
import pkg_resources
import platform
import subprocess
import sys

from setuptools import find_packages, setup


with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='audlib',
    version='0.0.3',
    author='Raymond Xia',
    author_email='raymondxia@cmu.edu',
    description='A speech signal processing library with emphasis on deep learning.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/raymondxyy/pyaudlib',
    download_url='https://github.com/raymondxyy/pyaudlib/archive/v_01.tar.gz',
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Multimedia :: Sound/Audio :: Analysis',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    keywords=['SPEECH', 'AUDIO', 'SIGNAL', 'SOUND', 'DEEP LEARNING', 'NEURAL NETWORKS'],
    license='MIT',
    install_requires=[
        'click >= 7.0',
        'numpy >= 1.17.2',
        'soundfile >= 0.10.2',
        'scipy >= 1.3.1',
        'resampy >= 0.2.2',
    ],
    entry_points={
        'console_scripts': [
            'audpipe = audlib.cli.audpipe:cli',
        ],
    },
    extras_require={
        'tests': [
            'pytest >= 5.1.3',
            'pytest-cov >= 2.8.1',
            'codecov >= 2.0.15',
        ],
        'nn': [
            'torch >= 1.2.0',
            'torchvision >= 0.4.0',
        ],
        'display': [
            'matplotlib >= 3.1.1',
        ],
    },
)
