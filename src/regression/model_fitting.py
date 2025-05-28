import pandas as pd
import json
import numpy as np
import optuna
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
from src.utils import ModelSaver


class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath

    def load_data(self):
        with self.filepath.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if 'Albums' in data and isinstance(data['Albums'], list):
            return pd.DataFrame(data['Albums'])
        else:
            logging.error("JSON structure invalid. 'Albums' key is missing or not a list.")
            return pd.DataFrame()


class FeatureExtractor:
    @staticmethod
    def avg_embedding(song_list):
        embeds = [d['audio_embedding'] for d in song_list]
        return np.mean(embeds, axis=0)

    def extract_features(self, df):
        X = np.vstack(df['songs'].apply(self.avg_embedding).values)
        y = df['score'].values
        return X, y


class CatBoostTrainer:
    def __init__(self, X, y):
        self.X_train, X_temp, self.y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=0)
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(X_temp, y_temp, test_size=0.5,
                                                                            random_state=0)
        self.model = CatBoostRegressor()

    def objective(self, trial):
        params = {
            "iterations": 500,
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "verbose": 0,
            "random_seed": 42
        }

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(self.model, self.X_test, self.y_test, cv=cv, scoring='neg_root_mean_squared_error')
        return scores.mean()

    def optimize(self, n_trials=30):
        study = optuna.create_study(direction="maximize", study_name="CatBoost Optimization")
        study.optimize(self.objective, n_trials=n_trials)
        return study.best_trial.params

    def train_final_model(self, best_params):
        best_params.update({
            "iterations": 1000,
            "early_stopping_rounds": 50,
            "eval_metric": "RMSE",
            "verbose": 100,
            "random_seed": 42
        })
        self.model = CatBoostRegressor(**best_params)
        self.model.fit(self.X_train, self.y_train, eval_set=(self.X_val, self.y_val))

    def evaluate(self):
        y_pred = self.model.predict(self.X_test)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        print(f"Final RMSE on test set: {rmse:.2f}")

    def get_model(self):
        return self.model


class Pipeline:
    def __init__(self, data_path):
        self.data_path = data_path

    def run(self):
        loader = DataLoader(self.data_path)
        df = loader.load_data()
        if df.empty:
            return

        extractor = FeatureExtractor()
        X, y = extractor.extract_features(df)

        trainer = CatBoostTrainer(X, y)
        best_params = trainer.optimize()
        trainer.train_final_model(best_params)
        trainer.evaluate()

        model = trainer.get_model()
        saver = ModelSaver(model, self.data_path)
        saver.save()


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[2]
    file_path = ROOT / "data" / "processed" / "dp.json"

    pipeline = Pipeline(file_path)
    pipeline.run()
