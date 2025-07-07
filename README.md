# Text-based approaches for exploring job matching techniques


This notebook implements and evaluates multiple text-based similarity models to text techniques of semantic matching between resumes and job descriptions. The findings presented in the capstone paper are directly tied to the code and results shown here.

## Project Structure

This project contains three main notebooks:

- `resume clean.ipynb`: Loads and preprocesses resume data (lowercasing, stopword removal, and tokenization).
- `jd clean.ipynb`: Performs similar preprocessing on job description texts.
- `capstone.ipynb`: Main analysis notebook that compares multiple semantic matching techniques.

## Datasets

- `Resume-clean.csv`: Preprocessed resumes, originally retrieved from https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset

- `Job-postings-10k.csv`: Preprocessed job descriptions, , originally retrieved from https://www.kaggle.com/datasets/arshkon/linkedin-job-postings


## Main Analysis

This notebook includes:
- **TF-IDF**
- **Word2Vec**
- **GloVe**
- **FastText**
- **BERT**

For various the models we take:
- **Similarity Scores** between resumes and job descriptions.
- **Cluster Quality** using silhouette scores.
- **Visualizations** such as similarity heatmaps and category-based groupings.

These datasets are loaded and used across the experiments in the main notebook.

Each major result discussed in the final paper is traceable to a labeled code section in the `capstone.ipynb` notebook.

## Dependencies

This project uses:
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`, `seaborn`
- `nltk`
- `transformers`

Ensure you have these libraries installed before running the notebooks.

Some models are loaded via the `gensim.downloader` API. You can download and use the required pre-trained embeddings as follows:

```python
import gensim.downloader as api

fasttext_model = api.load("fasttext-wiki-news-subwords-300")
glove_model = api.load("glove-wiki-gigaword-50")
```
Ensure you have these libraries installed before running the notebooks.

## Summary

This README links the codebase with the academic findings by outlining the steps taken and highlighting how each embedding model contributes to the overall comparison of semantic job matching strategies.
