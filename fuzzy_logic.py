import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# Define universe of discourse for sentiment_score (from -1 to 1)
x_sentiment = np.arange(-1, 1.01, 0.01)

# Define fuzzy membership functions for sentiment
neg = fuzz.trimf(x_sentiment, [-1, -1, 0])
neu = fuzz.trimf(x_sentiment, [-0.2, 0, 0.2])
pos = fuzz.trimf(x_sentiment, [0, 1, 1])

# Plot membership functions (optional)
plt.figure(figsize=(8, 4))
plt.plot(x_sentiment, neg, label="Negative")
plt.plot(x_sentiment, neu, label="Neutral")
plt.plot(x_sentiment, pos, label="Positive")
plt.title("Fuzzy Membership Functions for sentiment_score")
plt.xlabel("sentiment_score")
plt.ylabel("Membership degree")
plt.legend()
plt.show()


# Function to fuzzify a sentiment score
def fuzzify_sentiment(score):
    neg_degree = fuzz.interp_membership(x_sentiment, neg, score)
    neu_degree = fuzz.interp_membership(x_sentiment, neu, score)
    pos_degree = fuzz.interp_membership(x_sentiment, pos, score)

    degrees = {"Negative": neg_degree, "Neutral": neu_degree, "Positive": pos_degree}

    # Find the label with the highest membership degree
    fuzzy_label = max(degrees, key=degrees.get)
    return fuzzy_label, degrees


# Example usage
score = 0.3
label, degrees = fuzzify_sentiment(score)
print(f"Sentiment score: {score} → Fuzzy label: {label}, Membership degrees: {degrees}")
