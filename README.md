# audlib

[![PyPI version](https://badge.fury.io/py/audlib.svg)](https://badge.fury.io/py/audlib)
[![Build Status](https://travis-ci.com/raymondxyy/pyaudlib.svg?token=xNuzdfgseSXz1yHDnh9L&branch=master)](https://travis-ci.org/raymondxyy/pyaudlib)
[![Coverage](https://codecov.io/gh/raymondxyy/pyaudlib/branch/master/graph/badge.svg?token=vMLw7Y9H5m)](https://codecov.io/gh/raymondxyy/pyaudlib)

> A speech signal processing library in Python with emphasis on deep learning.

audlib provides a collection of utilities for developing speech-related applications using both signal processing and deep learning. The package offers the following high-level features:

- Speech signal processing utilities with ready-to-use applications
- Deep learning architectures for speech processing tasks in [PyTorch][pytorch]
- PyTorch-compatible interface (similar to torchvision) for batch processing
- A command-line interface with a unix-pipe-like syntax

Some use cases of audlib are:

- Extracting common speech features for your backend
- Developing your own deep-learning-based tools for speech tasks
- Quickly try out speech processors and visualize the spectrogram in command line

audlib focuses on correctness, efficiency, and simplicity. Signal processing functionalities are mathematically checked whenever possible (e.g. constant overlap-add, `istft(stft(X))==X`). Deep neural networks follow the [PyTorch][pytorch]'s convention.

## Breaking Changes

- 0.0.3
  - `sig.util.freqz` becomes `sig.util.fftfreqz`
  - `nn.strfnet` is removed. See the official repository [here](https://github.com/raymondxyy/strfnet-IS2020).
  - `transform.stlogm` is removed
- 0.0.2
  - `audioread` follows the interface of `soundfile.read`
  - `audiowrite` follows the interface of `soundfile.write`
  - The argument `sr` is removed from all short-time transforms

## Installation

```sh
pip install audlib
```

## Developer Installation

In the source directory, install the library with test dependencies:

```sh
pip install ".[tests]"
```

Run test:

```sh
python -m pytest tests
```

## Release flow

1. Bump version in setup.py.
2. Package release: `python setup.py sdist bdist_wheel`
3. Upload release: `twine upload --repository-url https://upload.pypi.org/legacy/ dist/*`

## Usage example

More extensive examples can be found in `examples/`.

## Release history

- 0.0.3
  - First release of the command-line tool *audpipe*
- 0.0.2
  - Streamlines optional installation
  - Improves API (**see breaking changes**)
  - Adds coverage test
- 0.0.1
  - First release on PyPI

## Contact 

Please reach out to Raymond Xia (raymondxia@cmu.edu) for comments.

More information about our research groups can be found on at

- [CMU Robust Speech Recognition Group](http://www.cs.cmu.edu/~robust/)
- [Machine Learning for Signal Processing Group](http://mlsp.cs.cmu.edu/)

## License

Distributed under the MIT license. See ``LICENSE`` for more information.

[pytorch]: https://pytorch.org/
