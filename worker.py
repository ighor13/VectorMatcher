import os
import sys
import traceback
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer

# --- НАСТРОЙКА ПУТЕЙ ---
if getattr(sys, 'frozen', False):
    base_dir = os.path.dirname(sys.executable)
else:
    base_dir = os.path.dirname(os.path.abspath(__file__))


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, names=["id", "data"])
    df["id"] = df["id"].fillna("").astype(str)
    df["data"] = df["data"].fillna("").astype(str).str.strip()
    return df


class Matcher:
    def __init__(self, model_path: str, use_gpu: bool = True, batch_size: int = 256):
        self.batch_size = batch_size
        self.device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
        torch.set_num_threads(os.cpu_count() or 1)

        # Пытаемся найти стандартную модель в кэше пользователя
        if model_path == "BAAI/bge-m3":
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            local_path = os.path.join(cache_dir, "models--BAAI--bge-m3", "snapshots")
            
            # Ищем папку внутри snapshots (там лежат файлы весов)
            if os.path.exists(local_path):
                # Берем последнюю (или единственную) папку с хешем
                snapshots = os.listdir(local_path)
                if snapshots:
                    model_path = os.path.join(local_path, snapshots[0])
                    print(f"DEBUG: Найдена локальная модель по пути: {model_path}")

        print(f"DEBUG: Попытка загрузить модель. Путь: '{model_path}'")
        
        try:
            # Если это не наш путь к папке, а строка (BAAI/bge-m3 или пользовательская),
            # SentenceTransformer попробует скачать её из интернета, если нет локально.
            self.model = SentenceTransformer(
                model_path, 
                device=self.device
            )
            print("DEBUG: Модель успешно загружена.")
        except Exception as e:
            print("DEBUG: ОШИБКА ПРИ ЗАГРУЗКЕ МОДЕЛИ!")
            traceback.print_exc()
            raise e

    def _encode(self, texts, progress_cb=None):
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

    def run(self, old_df, new_df, progress_cb=None, log_cb=None):
        def log(msg):
            if log_cb:
                log_cb(msg)

        log(f"Устройство: {self.device}")
        log(f"old: {len(old_df)} строк, new: {len(new_df)} строк")

        log("Эмбеддинги new...")
        new_emb = self._encode(new_df["data"].tolist())
        if progress_cb:
            progress_cb(30)

        log("Эмбеддинги old...")
        old_emb = self._encode(old_df["data"].tolist())
        if progress_cb:
            progress_cb(60)

        log("Поиск совпадений...")
        new_emb_t = new_emb.t()
        results = []
        old_len = len(old_emb)
        total_batches = (old_len + self.batch_size - 1) // self.batch_size

        for bi, i in enumerate(range(0, old_len, self.batch_size)):
            batch = old_emb[i:i + self.batch_size]
            sim = torch.matmul(batch, new_emb_t)
            scores, idxs = torch.max(sim, dim=1)

            scores_np = scores.detach().cpu().numpy()
            idxs_np = idxs.detach().cpu().numpy()

            for k in range(len(scores_np)):
                oi = i + k
                ni = int(idxs_np[k])
                results.append({
                    "old_id": old_df.iloc[oi]["id"],
                    "old_data": old_df.iloc[oi]["data"],
                    "new_id": new_df.iloc[ni]["id"],
                    "new_data": new_df.iloc[ni]["data"],
                    "score": round(float(scores_np[k]), 4),
                })

            if progress_cb:
                progress_cb(60 + int(40 * (bi + 1) / total_batches))

        log("Готово.")
        return pd.DataFrame(results)