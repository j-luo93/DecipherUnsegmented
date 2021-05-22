# DecipherUnsegmented
This hosts the code for the paper, based on https://github.com/j-luo93/xib.

# Clean-up plan
* [x] phonological embedding composition
* [x] modules to compose embeddings
* [x] demo script for composing embeddings
* [x] pretrained phonological embedding
* [ ] core IPA helper modules
* [ ] core aligned corpus helper modules
* [ ] core training and model modules
* [ ] notebooks and scripts

# Install
* Clone this repository recursively `git clone --recursive <link>`.
* `pip install -r requirements.txt` to install Python dependencies
* Install `pytorch`
* Install `dev_misc` by running `cd dev_misc & pip install -e .`
* Install this repository by running `pip install -e .` in the root directory

# Run
The main repo is still being cleaned up, but you can see `scripts/demo_compose.py` for a demo script to compose phonological embeddings
based on IPA transcriptions.
