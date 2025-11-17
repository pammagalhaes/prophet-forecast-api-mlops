from pathlib import Path

# Caminho base do projeto (raiz)
BASE_DIR = Path(__file__).resolve().parents[1]  # raiz do projeto (ajustado)

# Caminhos dos dados
DATA_DIR = BASE_DIR / "data" / "raw"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
STORE_CSV = DATA_DIR / "store.csv"

# Caminho dos modelos
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


# split date for train/validation (example; adjust if needed)
TRAIN_TEST_SPLIT_DATE = "2015-06-01"
