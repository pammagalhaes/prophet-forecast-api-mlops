import pytest
from pathlib import Path
from src.modeling.model_utils import save_model, load_model
from prophet import Prophet
import tempfile

def test_save_and_load_model():
    model = Prophet()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # salvar
        saved_path = save_model(model, store_id=1, model_dir=tmp_path)
        assert saved_path.exists()

        # carregar
        loaded_model = load_model(store_id=1, model_dir=tmp_path)
        assert isinstance(loaded_model, Prophet)
