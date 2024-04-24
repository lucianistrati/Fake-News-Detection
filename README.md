# Fake-News-Detection

This repository contains a Python script for detecting fake news. The script uses machine learning techniques to classify news articles as either fake or real based on their content.

## Usage

To use the fake news detection script:

1. Ensure you have all the necessary dependencies installed. You can install them using the provided `requirements.txt` file.

2. Run the `main.py` script. This script reads the training data from the `textract_train.csv` file, trains a machine learning model, and evaluates its performance.

3. The script outputs the best score achieved by the model and its training score.

## Files

- `main.py`: Python script for fake news detection.
- `textract_train.csv`: Training data containing labeled news articles.
- `textract_test.csv`: Test data containing unlabeled news articles.

## Requirements

Ensure you have the following dependencies installed:

- Python 3.x
- `numpy`
- `pandas`
- `nltk`
- `scikit-learn`
- `wordcloud`
- `matplotlib`
- `spacy`

You can install these dependencies using the provided `requirements.txt` file by running:

```bash
pip install -r requirements.txt
```

## How it Works

The script preprocesses the text data, extracts features from the text using the Bag-of-Words approach, and trains a machine learning model. The model is then evaluated using cross-validation to determine its performance.


[![Stand With Ukraine](https://raw.githubusercontent.com/vshymanskyy/StandWithUkraine/main/banner2-direct.svg)](https://stand-with-ukraine.pp.ua)
