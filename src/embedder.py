import numpy as np
import onnxruntime as ort
from pathlib import Path
from tokenizers import Tokenizer

# Automatically locate project root (parent directory of 'src/')
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "Xenova" / "all-MiniLM-L6-v2"


class Embedder:
    def __init__(self, path: str | Path = DEFAULT_MODEL_DIR):
        model_path = Path(path)

        # Fallback to absolute path relative to project root if relative path is passed
        if not model_path.is_absolute():
            model_path = PROJECT_ROOT / model_path

        tokenizer_file = model_path / "tokenizer.json"
        onnx_file = model_path / "model.onnx"

        if not tokenizer_file.exists():
            raise FileNotFoundError(f"Tokenizer not found at: {tokenizer_file}")
        if not onnx_file.exists():
            raise FileNotFoundError(f"ONNX model not found at: {onnx_file}")

        self.tokenizer = Tokenizer.from_file(str(tokenizer_file))
        self.session = ort.InferenceSession(
            str(onnx_file), providers=["CPUExecutionProvider"]
        )
        self.input_names = {inp.name for inp in self.session.get_inputs()}

    def encode(self, text: str, normalize: bool = True) -> np.ndarray:
        return self.encode_batch([text], normalize=normalize)[0]

    def encode_batch(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        self.tokenizer.enable_padding()
        encoded = self.tokenizer.encode_batch(texts)
        feed = {}

        if "input_ids" in self.input_names:
            feed["input_ids"] = np.array([e.ids for e in encoded], dtype=np.int64)
        if "attention_mask" in self.input_names:
            feed["attention_mask"] = np.array(
                [e.attention_mask for e in encoded], dtype=np.int64
            )
        if "token_type_ids" in self.input_names:
            feed["token_type_ids"] = np.array(
                [e.type_ids for e in encoded], dtype=np.int64
            )

        hidden = self.session.run(None, feed)[0]
        mask = feed["attention_mask"][..., None]
        pooled = (hidden * mask).sum(axis=1) / mask.sum(axis=1)

        if normalize:
            pooled = pooled / np.linalg.norm(pooled, axis=1, keepdims=True)

        return pooled