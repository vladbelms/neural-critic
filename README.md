# Neural Critic: Album Evaluator

![Alt text](swag-monkey.webp)


**Neural Critic** is an open-source Streamlit app that evaluates music albums through deep audio analysis and professional-grade scoring. Trained on Metacritic scores from professional music reviewers, the app simulates the critical insight of a seasoned reviewer to give albums an authentic, critic-style rating.

Leveraging LAION's CLAP (Contrastive Language-Audio Pretraining) model for extracting rich audio embeddings and a fine-tuned CatBoost regression model, Neural Critic offers an intuitive and scalable platform for music evaluation.

## Features

* **Audio Embedding Generation** — Generate high-dimensional audio embeddings with the LAION-CLAP model.
* **Critic-Level Scoring** — Predict a professional-style album score, trained on real Metacritic ratings.
* **Web Interface** — Upload songs and receive an expert-style album score instantly.
* **Training Pipeline** — Includes preprocessing, Optuna-based hyperparameter tuning, and model training.
* **Automatic Model Download** — Automatically retrieves required CLAP model checkpoints if not available locally.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/vladbelms/neural-critic.git
cd neural-critic
```

### 2. Set Up a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install poetry
poetry install --no-root
```

## Run the App

Launch the Streamlit interface:

```bash
streamlit run main.py
```

Upload songs files from an album, and Neural Critic will analyze the audio to produce a professional critic-style score based on patterns learned from Metacritic reviews.

## Model Training

Under `src/`, you'll find:

* `embeddings/` for generating CLAP-based audio embeddings
* `regression/` for fitting and tuning the CatBoost regression model
* `utils/` for helper functions including model saving

All training and inference logic is structured under `src/` for modularity and clarity.

## Contributing

We welcome contributions! Submit issues or pull requests to improve features, performance, or UX.

## License

MIT License

## Contact

Maintainer: [@vladbelms](https://github.com/vladbelms)
