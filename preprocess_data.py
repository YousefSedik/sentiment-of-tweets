from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re


def preprocess_data(df: pd.DataFrame):

    # print('sentiment' in df.columns)
    # Drop unnecessary columns
    df.drop(
        columns=[
            "Datetime",
            "Tweet Id",
            "Username",
            "Unnamed: 0",
        ],
        inplace=True,
    )

    # Drop duplicates
    print("Duplicates:", df.duplicated().sum())
    df.drop_duplicates(inplace=True)

    # Drop null values
    print("Null values:\n", df.isnull().sum())
    df.dropna(inplace=True)

    # Clean the text
    df["Text"] = df["Text"].apply(
        lambda x: re.sub(r"[@#]\w+", "", x).strip()
    )  # Remove @mentions and #hashtags
    print("After mentions/hashtags:\n", df["Text"].head())

    df["Text"] = df["Text"].apply(
        lambda x: re.sub(r"http\S+|www\S+|https\S+", "", x, flags=re.MULTILINE).strip()
    )  # Remove URLs
    df["Text"] = df["Text"].apply(
        lambda x: re.sub(r"[^\x00-\x7F]+", "", x).strip()
    )  # Remove emojis
    df["Text"] = df["Text"].apply(
        lambda x: re.sub(r"[^a-zA-Z0-9\s]", "", x).strip()
    )  # Remove special characters
    df["Text"] = df["Text"].apply(
        lambda x: re.sub(r"\s+", " ", x).strip()
    )  # Remove extra spaces

    # Show Data Sentiment and Emoji Distribution
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.countplot(x="sentiment", data=df)
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.subplot(1, 2, 2)
    sns.countplot(x="emotion", data=df)
    plt.title("Emotion Distribution")
    plt.xlabel("Emotion")
    plt.ylabel("Count")
    plt.show()

    df.drop(
        columns=[
            "emotion",
            "emotion_score",
        ],
        inplace=True,
    )
    le_sentiment = LabelEncoder()
    df["sentiment"] = le_sentiment.fit_transform(df["sentiment"])

    # show the distribution of the classes
    print("Class distribution:\n", df["sentiment"].value_counts())

    # Reset index
    df.reset_index(drop=True, inplace=True)

    # Save cleaned data
    df.to_csv("data/cleaned_data.csv", index=False)

    return df, le_sentiment


if __name__ == "__main__":
    # Load your dataset
    df = pd.read_csv("data/sentiment-emotion-tweets.csv")
    print(df.info)
    # Preprocess the data
    df, le_sentiment = preprocess_data(df)
    # Save label encoders
    le_sentiment.classes_.tofile("data/le_sentiment_classes.txt", sep="\n")
