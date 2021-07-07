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
* Pretrained phonological embeddings are available in `data/got.pretrained.pth`.
* See `scripts/demo_compose.py` for a demo script to compose phonological embeddings based on IPA transcriptions.
* See `scripts/demo_pretrained.py` for a demo script to load a pretrained embedding layer. Once you run
`python scripts/demo_pretrained.py`, you would obtain two files:
`data/segments.emb.tsv` and `data/segments.tsv`.
Go to the [embedding projector](https://projector.tensorflow.org/) to visualize the embeddings by uploading both files.

# Data
* Iberian data is included in the repo. Three files are included: the original `data/hesperia_epigraphy.csv` that contains the published data from [Hesperia](http://hesperia.ucm.es/en/proyecto_hesperia.php), a Jupyter notebook `notebooks/clean_iberian.ipynb` that I used to clean up the data, and finally the cleaned csv `data/iberian.csv`.
