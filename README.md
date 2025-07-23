# Important: Please use the code at [https://github.com/itrummer/thalamusdb](https://github.com/itrummer/thalamusdb) for the latest version!

## ThalamusDB: Answering Complex Queries with Natural Language Predicates on Multi-Modal Data

[Research Paper](https://dl.acm.org/doi/10.1145/3654989) [Demo Paper](https://dl.acm.org/doi/abs/10.1145/3555041.3589730) [Demo Video](https://youtu.be/wV9UhULhFg8)

> ThalamusDB supports SQL queries with natural language predicates on multi-modal data. Our data model extends the relational model and integrates multi-modal data, including visual, audio, and text data, as columns. Users can write SQL queries including predicates on multi-modal data, described in natural language.

<del>

## Quick Start (GUI using OpenAI GPT-4-Turbo)

```bash
# Tested on Python 3.9.6.
git clone https://github.com/saehanjo/thalamusdb.git
cd thalamusdb

# Create virtual environment.
python3 -m venv .venv
source .venv/bin/activate

# Download and unzip Craigslist image files: 186 MB
sudo pip install gdown
gdown 1iy_H4jjeDDnDHY1wi1l6A6THodB2ShFY
mv furniture_imgs.zip craigslist
unzip craigslist/furniture_imgs.zip -d craigslist

# Install requirements.
python3 -m pip install -U pip
pip install --upgrade wheel
pip install -r requirements.txt

# Run ThalamusDB GUI (replace [OPENAI_API_ACCESS_KEY] with your key).
streamlit run gui.py [OPENAI_API_ACCESS_KEY]
```


## Run Benchmarks

```bash
# First, install ThalamusDB and download Craigslist image files based on Quick Start.

# Download Netflix movie ratings csv file: 2.71 GB
gdown 11lfRbEtJbdot_G3Qlr3lb-85ZYTrbC3c

# To run the YouTube benchmark, download the raw waveforms of the AudioCaps dataset: https://audiocaps.github.io/
# Then, unzip files into train, test, and val folders in audiocaps/waveforms: 32.26 GB
# Otherwise, comment out the YouTube benchmark from benchmark.py in order to just run the other two benchmarks.

# Run all benchmarks.
mkdir log
python benchmark.py
```

Note that the current configuration uses audio and image models from HuggingFace for ease of installation: [CLAP](https://huggingface.co/docs/transformers/en/model_doc/clap) and [clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32). In order to install the models used in our experiments, please download and setup the following models: [Audio](https://github.com/akoepke/audio-retrieval-benchmark) and [CLIP ResNet50](https://github.com/openai/CLIP).


## How to Integrate New Datasets

There are two ways to integrate new datasets.

### Using Command-Line Tool.

Run `console.py` to create a new database. Use `CREATE TABLE` statements to create tables with columns of `IMAGE`, `TEXT`, and `INTEGER` data types (`AUDIO` data type is not yet supported on console). Add foreign key constraints by using the `ALTER TABLE` command. Use `COPY` statements to insert rows from a csv file. Run queries with `SELECT` statements with natural language predicates on multi-modal data using the `NL` keyword.

Example:
```sql
CREATE TABLE furniture(time INTEGER, neighborhood TEXT, title TEXT, url TEXT, price INTEGER, aid INTEGER);
CREATE TABLE images(img IMAGE, aid INTEGER);
ALTER TABLE images ADD FOREIGN KEY (aid) REFERENCES furniture (aid);
COPY furniture FROM 'craigslist/formated_furniture.csv' DELIMITER ',';
COPY images FROM 'craigslist/formated_imgs.csv' DELIMITER ',';
SELECT max(price) FROM images, furniture WHERE images.aid = furniture.aid AND nl(img, 'wooden');
```

### Using an NLDatabase instance.

Create a new function in `nldbs.py` that creates a NLDatabase instance (refer to functions `craisglist()`, `youtubeaudios()`, and `netflix()`). It requires loading relational data to DuckDB and providing pointers to image and audio data.
</del>
