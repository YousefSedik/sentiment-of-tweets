from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


class ANFIS:
    def __init__(self, n_membership=2, learning_rate=0.01, epochs=100, max_rules=32):
        self.n_membership = n_membership
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.max_rules = max_rules
        self.centers = None
        self.spreads = None
        self.consequent_params = None

    def triangular_membership(self, x, center, spread):
        """Triangular membership function"""
        left = center - spread
        right = center + spread
        return np.maximum(
            0, np.minimum((x - left) / (spread + 1e-15), (right - x) / (spread + 1e-15))
        )

    def layer1(self, X):
        """Fuzzification layer"""
        n_samples, n_features = X.shape
        memberships = np.zeros((n_samples, n_features, self.n_membership))

        for i in range(n_features):
            for j in range(self.n_membership):
                memberships[:, i, j] = self.triangular_membership(
                    X[:, i], self.centers[i, j], self.spreads[i, j]
                )
        return np.clip(memberships, 1e-10, 1.0)

    def layer2_3(self, memberships):
        """Simplified rule layer with limited rules"""
        n_samples = memberships.shape[0]
        n_features = memberships.shape[1]

        feature_memberships = np.mean(memberships, axis=2)

        firing_strengths = np.zeros((n_samples, self.max_rules))

        # For reproducibility
        np.random.seed(42)

        for i in range(self.max_rules):
            selected_features = np.random.choice(
                n_features, size=min(3, n_features), replace=False
            )
            firing_strengths[:, i] = np.mean(
                feature_memberships[:, selected_features], axis=1
            )

        """Normalize firing strengths"""
        sum_firing = np.sum(firing_strengths, axis=1, keepdims=True) + 1e-10
        return firing_strengths / sum_firing

    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -15, 15)))

    def forward_pass(self, X):
        """Forward pass with sigmoid activation"""
        memberships = self.layer1(X)
        normalized_firing_strengths = self.layer2_3(memberships)

        X_expanded = np.column_stack([X, np.ones(X.shape[0])])
        consequent_outputs = np.dot(X_expanded, self.consequent_params.T)
        weighted_sum = np.sum(normalized_firing_strengths * consequent_outputs, axis=1)

        return self.sigmoid(weighted_sum)

    def initialize_parameters(self, X):
        """Initialize parameters"""
        n_features = X.shape[1]

        self.centers = np.zeros((n_features, self.n_membership))
        self.spreads = np.zeros((n_features, self.n_membership))

        for i in range(n_features):
            feature_min, feature_max = np.min(X[:, i]), np.max(X[:, i])
            feature_range = feature_max - feature_min

            """ Initialize centers"""
            self.centers[i] = np.linspace(feature_min, feature_max, self.n_membership)

            """Initialize spreads"""
            self.spreads[i] = np.ones(self.n_membership) * (
                feature_range / (self.n_membership - 0.5)
            )

        """Initialize consequent parameters with limited rules"""
        self.consequent_params = np.random.normal(
            0, 0.1, (self.max_rules, n_features + 1)
        )

    def fit(self, X, y):
        """Train the model"""
        self.initialize_parameters(X)

        best_accuracy = 0
        best_params = None

        for epoch in range(self.epochs):
            predictions = self.forward_pass(X)
            binary_predictions = (predictions >= 0.5).astype(int)
            accuracy = accuracy_score(y, binary_predictions)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = (
                    self.centers.copy(),
                    self.spreads.copy(),
                    self.consequent_params.copy(),
                )

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Accuracy: {accuracy:.4f}")

            error = predictions - y

            X_expanded = np.column_stack([X, np.ones(X.shape[0])])
            memberships = self.layer1(X)
            normalized_firing_strengths = self.layer2_3(memberships)

            for i in range(self.max_rules):
                gradient = np.dot(
                    error
                    * normalized_firing_strengths[:, i]
                    * predictions
                    * (1 - predictions),
                    X_expanded,
                )
                self.consequent_params[i] -= self.learning_rate * gradient

            for i in range(X.shape[1]):
                for j in range(self.n_membership):
                    center_gradient = np.mean(
                        error
                        * normalized_firing_strengths[:, 0]
                        * (X[:, i] - self.centers[i, j])
                    )
                    self.centers[i, j] -= self.learning_rate * center_gradient

                    spread_gradient = np.mean(
                        error
                        * normalized_firing_strengths[:, 0]
                        * np.abs(X[:, i] - self.centers[i, j])
                    )
                    self.spreads[i, j] -= self.learning_rate * spread_gradient
                    self.spreads[i, j] = max(self.spreads[i, j], 1e-5)

        self.centers, self.spreads, self.consequent_params = best_params

    def predict(self, X):
        """Predict binary classes"""
        predictions = self.forward_pass(X)
        return (predictions >= 0.5).astype(int)

    def predict_proba(self, X):
        """Predict probabilities"""
        return self.forward_pass(X)


class MulticlassANFIS:
    """Extension of ANFIS to handle multiclass classification using one-vs-rest strategy"""

    def __init__(
        self, n_classes, n_membership=2, learning_rate=0.01, epochs=100, max_rules=32
    ):
        self.n_classes = n_classes
        self.models = []
        for _ in range(n_classes):
            self.models.append(ANFIS(n_membership, learning_rate, epochs, max_rules))

    def fit(self, X, y):
        """Train one model per class"""
        for class_idx in range(self.n_classes):
            print(f"\nTraining model for class {class_idx}")
            # Create binary target (1 for current class, 0 for all others)
            binary_target = (y == class_idx).astype(int)
            self.models[class_idx].fit(X, binary_target)

    def predict_proba(self, X):
        """Predict probabilities for each class"""
        probs = np.zeros((X.shape[0], self.n_classes))
        for class_idx in range(self.n_classes):
            probs[:, class_idx] = self.models[class_idx].predict_proba(X)

        # Normalize probabilities to sum to 1
        row_sums = probs.sum(axis=1, keepdims=True)
        return probs / row_sums

    def predict(self, X):
        """Predict class with highest probability"""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


# Load the data
def load_data():
    try:
        # Load text embeddings
        embeddings = np.load("data/embeddings.npy")

        # Load sentiment data
        df = pd.read_csv("data/cleaned_data.csv")

        print(
            f"Loaded {embeddings.shape[0]} embeddings with dimension {embeddings.shape[1]}"
        )
        print(f"Loaded {df.shape[0]} records from CSV with {df.shape[1]} columns")

        return embeddings, df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None


def main():
    # Load data
    embeddings, df = load_data()

    # Get the target variable (sentiment)
    y = df["sentiment"].values

    # Use the embeddings as features (X)
    X = embeddings
    print(f"Feature shape: {X.shape}")
    print(X)
    # Check how many unique sentiments we have
    n_sentiments = len(np.unique(y))
    
    # Scale the features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")

    # Initialize and train the multiclass ANFIS model
    # Reduce the dimensionality of input for efficiency if needed
    max_features = min(50, X_train.shape[1])  # Use at most 50 features
    if X_train.shape[1] > max_features:
        print(
            f"Reducing feature dimensionality from {X_train.shape[1]} to {max_features}"
        )
        from sklearn.decomposition import PCA

        pca = PCA(n_components=max_features)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    # Train model with reduced parameters for efficiency
    model = MulticlassANFIS(
        n_classes=n_sentiments,
        n_membership=2,  # Fewer membership functions
        learning_rate=0.01,
        epochs=20,  # Fewer epochs
        max_rules=16,  # Fewer rules
    )

    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy * 100:.2f}%")



if __name__ == "__main__":
    main()
