# pyaudlib

[![Build Status](https://travis-ci.com/raymondxyy/pyaudlib.svg?token=xNuzdfgseSXz1yHDnh9L&branch=master)](https://travis-ci.org/raymondxyy/pyaudlib)

> A speech signal processing library in Python with emphasis on deep learning.

pyaudlib (name subject to change) provides a collection of utilities for developing speech-related applications using both signal processing and deep learning. The package offers the following high-level features:

- Speech signal processing utilities with ready-to-use applications
- Deep learning architectures for speech processing tasks in [PyTorch][pytorch]
- PyTorch-compatible interface (similar to torchvision) for batch processing
- I/O utilities for interfacing with [CMUSPHINX][sphinx]
- A command-line interface with a unix-pipe-like syntax

Some use cases of pyaudlib are:

- Extracting common speech features for your backend
- Integrating CMUSPHINX with modern deep learning architectures
- Developing your own deep-learning-based tools for speech tasks
- Quickly try out speech enhancers and visualize the spectrogram in command line

pyaudlib focuses on correctness, efficiency, and simplicity. Signal processing functionalities are mathematically checked whenever possible (e.g. constant overlap-add, `istft(stft(X))==X`), and benchmarked in comparison to popular speech or audio processing toolbox (e.g. librosa). Deep neural networks and the training pipeline are consistent with that of PyTorch and torchvision.

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
python -m pytest tests/sig
```

## Release flow
1. Bump version in setup.py.
2. Package release: `python setup.py sdist bdist_wheel`
3. Upload release: `twine upload --repository-url https://upload.pypi.org/legacy/ dist/*`

## Usage example

- Command-line feature extraction and visualization
```sh
audiopipe open -i samples/welcome16k.wav read logspec plot
```
![](https://filedn.com/lx3TheNX5ifLtAEMJg2YxFh/sn/pyaudlib/welcome-logspec.png)

More extensive examples can be found in `egs`.


## Development setup

See `doc/0-getting-started.md`.

## Release history

- 0.0.1
    - Work in progress

## Authors

Raymond Xia - yangyanx@andrew.cmu.edu

Mahmoud Alismail - mahmoudi@andrew.cmu.edu

Shangwu Yao - shangwuyao@gmail.com

Feel free to send us any issue you find and question you have.

## Contributing

Please contact one of the authors.

## License
Distributed under the GNU GPLv3 license. See ``LICENSE`` for more information.

[pytorch]: https://pytorch.org/
[sphinx]: https://cmusphinx.github.io/
