"""Read-only local-cache diagnosis of embedding position buffers; no provider API."""
import json
import os
from pathlib import Path
import sys
os.environ["HF_HUB_OFFLINE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import load_settings
from sentence_transformers import SentenceTransformer
import torch

settings = load_settings()
model = SentenceTransformer(settings.memory_embedding_model, device="cpu", cache_folder=str(settings.memory_model_cache_dir), trust_remote_code=True, local_files_only=True)
model.max_seq_length = 1024
result = {"device": "cpu", "model": settings.memory_embedding_model, "buffers": []}
for name, tensor in model.named_buffers():
    if "position_ids" in name:
        value = tensor.detach().cpu()
        result["buffers"].append({"name": name, "shape": list(value.shape), "first": value.flatten()[:20].tolist(), "min": value.min().item(), "max": value.max().item(), "is_arange": bool(torch.equal(value.flatten(), torch.arange(value.numel(), dtype=value.dtype)))})
target = Path(sys.argv[1])
target.write_text(json.dumps(result, indent=2), encoding="utf-8")
try:
    vector = model.encode(["测试短句"], normalize_embeddings=True)
    result["short_encode"] = {"shape": list(vector.shape), "finite": bool(torch.tensor(vector).isfinite().all())}
except Exception as error:
    result["short_encode"] = {"error": repr(error)}
target.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result))

# A proposed compatibility repair, confined to this diagnostic process. No model
# cache, production source, or persistent model weights are changed.
for name, module in model.named_modules():
    if hasattr(module, "position_ids") and name.endswith("embeddings"):
        old = module.position_ids
        module.register_buffer("position_ids", torch.arange(old.numel(), dtype=old.dtype, device=old.device).reshape(old.shape), persistent=False)
try:
    texts = ["测试短句", "Python CSV 数据转换与 JSON 验证", "长文本边界测试" * 300]
    vectors = model.encode(texts, normalize_embeddings=True)
    result["diagnostic_only_repair"] = {"shape": list(vectors.shape), "finite": bool(torch.tensor(vectors).isfinite().all()), "norms": torch.tensor(vectors).norm(dim=1).tolist()}
except Exception as error:
    result["diagnostic_only_repair"] = {"error": repr(error)}
target.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result["diagnostic_only_repair"]))
