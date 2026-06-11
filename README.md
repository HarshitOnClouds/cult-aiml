# Netflix Prize Recommendation System

A complete, end-to-end personalized recommendation system using the Netflix Prize Dataset, built for an ML competition.

## Features
- **Data Pipeline:** Efficient parsing of 100M+ ratings into Parquet format, stratified sampling into Tiny/Medium subsets, and generation of Sparse Interaction Matrices.
- **Models:** 
  - User-Based Collaborative Filtering
  - Item-Based Collaborative Filtering
  - Matrix Factorization via SVD (`scikit-surprise`)
  - Alternating Least Squares (ALS) (`implicit`)
- **Evaluation:** Custom implementations of MAP@10 and RMSE to correctly benchmark the models according to competition rules.
- **Generators:** Top-K recommendation generation with item-based explainability.

## Setup Instructions

### 1. Requirements
Ensure you have Python 3.10+ installed.
We recommend using a virtual environment.

```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Dataset
Download the [Netflix Prize Data from Kaggle](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data) and place the extracted files inside the `data/raw/` directory:
- `combined_data_1.txt` ... `combined_data_4.txt`
- `movie_titles.csv`

### 3. Running the Data Pipeline
Run the preprocessing pipeline to parse the dataset, create subsets, and build sparse matrices:

```bash
python src/data/loader.py
python src/data/sampler.py
python src/data/preprocessor.py
```
*Note: The `loader.py` script parses ~2GB of text into a Pandas DataFrame. It may take 5-15 minutes and requires roughly 4GB of free RAM.*

### 4. Running the Models
You can run the full end-to-end benchmark script on the generated "Tiny" subset to test everything:

```bash
python main.py
```

### 5. Notebooks
Explore the `notebooks/` directory to see:
- `01_eda.ipynb`: Exploratory Data Analysis
- `06_evaluation.ipynb`: Deep dive into model comparison and Top-K generation

## Repository Structure
- `data/`: Raw data, processed parquet files, and sampled subsets
- `notebooks/`: Jupyter Notebooks for EDA and Evaluation
- `src/`: Core Python modules (data, models, evaluation, recommendation)
- `reports/`: Deliverables including PDF reports and figures
- `main.py`: Entry point for quick experiment running
