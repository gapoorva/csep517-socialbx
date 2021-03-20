# socialbx project

This project was done for CSEP 517 Natural Language Processing at the University of Washington as part of my Profesional Master's Program.

## Installation

This project was built using cuda. Do the following to create a cuda setup.

```bash
# Create a conda environment
conda create --name NLPCourse

# Activate the conda environment
conda activate NLPCourse

# Install dependencies
conda install -c conda-forge spacy nltk keras tensorflow pandas scikit-learn gensim numpy

# Add socialbx to your python path
export PYTHONPATH=PYTHONPATH:$(dirname $(pwd))
```

## Usage

```bash
# Train a the model
python sentiment/train.py

# Evaluate the model on the test set
python sentiment/evaluate.py

# Test the model preditions on `test/sentiment-inference.test.csv`
python sentiment/predict.py test/sentiment-inference.test.csv

# Run social BX analysis on a pretrained model and using spaCy for NER
python analysis.py
```