import os

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from sklearn.model_selection import train_test_split
from keras.api.layers import Input, Dense, Dropout
from keras.api.utils import to_categorical
from keras.api.models import Model
import pandas as pd
import numpy as np

# Load your dataset
df = pd.read_csv("data/cleaned_data.csv")

# Load embeddings
X = np.load("data/embeddings.npy")  # Shape: (24970, embedding_dim)

y_sentiment = df["sentiment"].values
y_emotion = df["emotion"].values

y_sentiment_cat = to_categorical(y_sentiment)
y_emotion_cat = to_categorical(y_emotion)

X_train, X_test, y_train_sent, y_test_sent = train_test_split(
    X, y_sentiment_cat, test_size=0.2, random_state=42
)

_, _, y_train_emo, y_test_emo = train_test_split(
    X, y_emotion_cat, test_size=0.2, random_state=42
)


input_dim = X.shape[1]
num_classes_sent = y_sentiment_cat.shape[1]
print("Input dimension:", input_dim, "Output dimension:", num_classes_sent)
input_layer = Input(shape=(input_dim,))
x = Dense(128, activation="relu")(input_layer)
x = Dropout(0.3)(x)
x = Dense(64, activation="relu")(x)
output = Dense(num_classes_sent, activation="softmax")(x)

model_sent = Model(inputs=input_layer, outputs=output)
model_sent.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)

model_sent.summary()
model_sent.fit(X_train, y_train_sent, validation_split=0.1, epochs=10, batch_size=64)

# Evaluate the model

y_pred_sent = model_sent.predict(X_test)
y_pred_sent_labels = np.argmax(y_pred_sent, axis=1)
y_true_sent_labels = np.argmax(y_test_sent, axis=1)

accuracy = model_sent.evaluate(X_test, y_test_sent, verbose=0)[1]
print(f"Sentiment Classification Accuracy: {accuracy:.4f}")

model_sent.save("model.keras")
