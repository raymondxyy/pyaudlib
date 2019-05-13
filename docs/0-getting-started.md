## Getting Started with `audlib`

After cloning the repository to your working directory, Do the following.

0. Make sure you're using Python 3 (preferably >=3.6). Python 2 is not supported and is not in the plan (community support will end in 2020 anyway). Make sure pip is installed.

1. Install pipenv with `pip install pipenv`.

2. cd into pyaudlib/ and create the required environment with `pipenv install --dev`. After installation, type `pipenv graph` and make sure you see a list that has the following packages in bold before proceeding:

    - Click
	- matplotlib
	- pytest
	- resampy
	- SoundFile

3. Activate the virtual environment with `pipenv shell`. You should see (pyaudlib) appearing at the beginning of the command prompt now.

4. cd into pyaudlib/tools/sph2pipe/ and compile the package with `gcc -o sph2pipe *.c -lm`. Make sure the compiled binary `sph2pipe` is executable by calling `./sph2pipe`. A help message should appear.

5. Add pyaudlib/ (full path) to `PYTHONPATH` in profiles such as `~/.bashrc` and source it with `. ~/.bashrc`.

6. cd back to pyaudlib/ and verify the package with `pytest tests/sig`. If the test is passed, you're good to go. (NOTE: normally pytest without any argument should just work, but I'm having a bug on matplotlib's backend at the moment on burro.)

---
Created 11/30/2018, 5:53:54 PM
Updated 12/1/2018, 5:32:51 PM
