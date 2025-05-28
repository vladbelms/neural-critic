from pathlib import Path
from catboost import CatBoostRegressor


class ModelSaver:
    def __init__(self, model, reference_file, save_dir ="models", filename="catboost_model.cbm"):
        self.model = model
        self.save_path = Path(reference_file).parent / save_dir / filename

    def save(self):
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(self.save_path))
        print(f"Model saved to {self.save_path}")

    @staticmethod
    def load(path):
        model = CatBoostRegressor()
        model.load_model(str(path))
        return model
