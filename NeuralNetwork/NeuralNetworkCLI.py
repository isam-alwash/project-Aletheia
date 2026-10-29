import json
import os
import glob
from datetime import datetime
import numpy as np

# =====================================================================
# CONFIGURATION & HYPERPARAMETERS
# =====================================================================
INPUT_SIZE = 5                  # Adjust to your feature count
OUTPUT_SIZE = 3                 # Adjust to your target output classes
HIDDEN_LAYERS = [16, 8]         # Neurons per hidden layer

EPOCHS = 10000                  # Training iterations
LEARNING_RATE = 0.05            # Gradient descent step size
TEST_RATIO = 0.2                # 20% validation split
RANDOM_SEED = 42                # Seed for reproducible initialization

WEIGHTS_FILE = "model_weights.json"
BACKUP_DIR = "backups"

# =====================================================================
# GENERAL PURPOSE NEURAL NETWORK ENGINE
# =====================================================================
class NeuralNetwork:
    """
    A fully configurable, pure NumPy feedforward neural network 
    with dynamic layer structure, backpropagation, and persistent weight storage.
    """
    def __init__(self, input_dim=INPUT_SIZE, hidden_layers=HIDDEN_LAYERS, output_dim=OUTPUT_SIZE,
                 lr=LEARNING_RATE, weights_file=WEIGHTS_FILE, backup_dir=BACKUP_DIR):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.lr = lr
        self.weights_file = weights_file
        self.backup_dir = backup_dir
        
        # Build network topology: [Input, Hidden_1, ..., Hidden_N, Output]
        self.layer_sizes = [self.input_dim] + self.hidden_layers + [self.output_dim]
        self.weights, self.biases = self.load_or_initialize_weights()

    # --- Activation Functions ---
    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clipped to prevent overflow

    @staticmethod
    def _d_sigmoid(x):
        return x * (1 - x)

    # --- Weight Persistence & Backups ---
    def load_or_initialize_weights(self, filepath=None):
        target_file = filepath if filepath else self.weights_file
        
        if os.path.exists(target_file):
            with open(target_file, "r") as f:
                data = json.load(f)
            weights = [np.array(w) for w in data["weights"]]
            biases = [np.array(b) for b in data["biases"]]
            print(f"[+] Loaded weights from '{target_file}' (Saved: {data.get('timestamp', 'N/A')})")
            return weights, biases

        print(f"[!] '{target_file}' not found. Initializing new parameters...")
        np.random.seed(RANDOM_SEED)
        weights = [np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) * 0.1 
                   for i in range(len(self.layer_sizes) - 1)]
        biases = [np.zeros((1, self.layer_sizes[i+1])) for i in range(len(self.layer_sizes) - 1)]
        
        self.save_weights(weights, biases)
        return weights, biases

    def save_weights(self, weights=None, biases=None):
        weights = weights if weights is not None else self.weights
        biases = biases if biases is not None else self.biases
        
        payload = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "architecture": self.layer_sizes,
            "weights": [w.tolist() for w in weights],
            "biases": [b.tolist() for b in biases]
        }
        
        # Save to main file
        with open(self.weights_file, "w") as f:
            json.dump(payload, f, indent=4)
            
        # Create timestamped backup
        os.makedirs(self.backup_dir, exist_ok=True)
        backup_path = os.path.join(self.backup_dir, f"weights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(backup_path, "w") as f:
            json.dump(payload, f, indent=4)
            
        print(f"[+] Saved weights to '{self.weights_file}' and backup: '{backup_path}'")

    def rollback(self, backup_filepath):
        """Restores model parameters from a specific backup file."""
        self.weights, self.biases = self.load_or_initialize_weights(filepath=backup_filepath)
        self.save_weights()
        print(f"[+] Model rolled back to '{backup_filepath}'")

    # --- Forward & Backward Pass ---
    def forward(self, X):
        activations = [X]
        layer_input = X
        
        # Hidden layers pass
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            layer_input = self._sigmoid(np.dot(layer_input, w) + b)
            activations.append(layer_input)
            
        # Output layer pass
        output = self._sigmoid(np.dot(layer_input, self.weights[-1]) + self.biases[-1])
        activations.append(output)
        return activations

    def train(self, X_train, Y_train, X_val=None, Y_val=None, epochs=EPOCHS):
        print("\n--- Training Started ---")
        for epoch in range(1, epochs + 1):
            activations = self.forward(X_train)
            
            # Backpropagation
            error = Y_train - activations[-1]
            delta = error * self._d_sigmoid(activations[-1])
            deltas = [delta]
            
            for i in range(len(self.weights) - 2, -1, -1):
                delta = deltas[-1].dot(self.weights[i+1].T) * self._d_sigmoid(activations[i+1])
                deltas.append(delta)
            deltas.reverse()

            # Weight and Bias Updates
            for i in range(len(self.weights)):
                self.weights[i] += self.lr * activations[i].T.dot(deltas[i])
                self.biases[i] += self.lr * np.sum(deltas[i], axis=0, keepdims=True)

            # Progress Logging
            if epoch % (epochs // 5) == 0 or epoch == epochs:
                tr_acc, tr_loss = self.evaluate(X_train, Y_train)
                val_info = ""
                if X_val is not None and Y_val is not None:
                    v_acc, v_loss = self.evaluate(X_val, Y_val)
                    val_info = f" | Val Loss: {v_loss:.5f} (Acc: {v_acc:.1f}%)"
                print(f"Epoch {epoch:5d}/{epochs} | Train Loss: {tr_loss:.5f} (Acc: {tr_acc:.1f}%){val_info}")

        self.save_weights()

    def evaluate(self, X, Y):
        """Calculates accuracy and Mean Absolute Error (MAE)."""
        if len(X) == 0:
            return 0.0, 0.0
        preds = self.forward(X)[-1]
        loss = np.mean(np.abs(Y - preds))
        
        # Classification evaluation (Argmax comparison)
        if Y.ndim > 1 and Y.shape[1] > 1:
            acc = np.mean(np.argmax(preds, axis=1) == np.argmax(Y, axis=1)) * 100.0
        else:
            acc = np.mean(np.round(preds) == Y) * 100.0
            
        return acc, loss

    def predict(self, X):
        """Returns raw output activations for new inputs."""
        return self.forward(X)[-1]


# =====================================================================
# DATASET HELPERS
# =====================================================================
def preprocess_dataset(X, Y, test_ratio=TEST_RATIO, seed=RANDOM_SEED):
    """
    Min-Max normalizes feature values to range [0, 1] 
    and splits into Training and Validation sets.
    """
    min_v, max_v = np.min(X, axis=0), np.max(X, axis=0)
    denom = np.where((max_v - min_v) == 0, 1, max_v - min_v)
    X_norm = (X - min_v) / denom

    np.random.seed(seed)
    idx = np.arange(len(X_norm))
    np.random.shuffle(idx)
    
    split = int(len(X_norm) * (1 - test_ratio))
    return X_norm[idx[:split]], X_norm[idx[split:]], Y[idx[:split]], Y[idx[split:]], min_v, max_v


# =====================================================================
# USAGE EXAMPLE (REPLACE WITH YOUR DATA IN YOUR OWN APP)
# =====================================================================
if __name__ == '__main__':
    # 1. Define arbitrary sample dataset (N samples x 5 features)
    dummy_X = np.random.uniform(0, 100, size=(20, INPUT_SIZE))
    # Target outputs (N samples x 3 target classes - One-Hot Encoded)
    dummy_Y = np.eye(OUTPUT_SIZE)[np.random.choice(OUTPUT_SIZE, size=20)]

    # 2. Preprocess Data
    X_tr, X_val, Y_tr, Y_val, min_vals, max_vals = preprocess_dataset(dummy_X, dummy_Y)

    # 3. Initialize Blank Engine
    model = NeuralNetwork(
        input_dim=INPUT_SIZE,
        hidden_layers=HIDDEN_LAYERS,
        output_dim=OUTPUT_SIZE,
        lr=LEARNING_RATE
    )

    # 4. Train Model
    model.train(X_tr, Y_tr, X_val, Y_val, epochs=EPOCHS)

    # 5. Predict on New Raw Sample
    sample_raw = np.array([[15.0, 32.1, 8.5, 42.0, 10.0]])
    sample_norm = (sample_raw - min_vals) / np.where((max_vals - min_vals) == 0, 1, max_vals - min_vals)
    
    predictions = model.predict(sample_norm)
    print("\n--- Inference ---")
    print("Raw Predictions:", predictions)
    print("Predicted Class Index:", np.argmax(predictions))
