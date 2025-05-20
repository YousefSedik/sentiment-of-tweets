from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = pd.read_csv(r"data/cleaned_data.csv")["Text"].tolist()

embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)

np.save("data/embeddings.npy", embeddings)
