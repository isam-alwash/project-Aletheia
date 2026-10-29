import json
import os
import glob
from datetime import datetime
import numpy as np

# =====================================================================
# CONFIGURATION & HYPERPARAMETERS
# =====================================================================
INPUT_SIZE = 5            # Features: voltage, current, temperature, humidity, wind_speed
OUTPUT_SIZE = 3           # Classes: Battery [1,0,0], Solar Panel [0,1,0], Grid [0,0,1]
HIDDEN_LAYERS = [10, 8, 6, 4]

EPOCHS = 10000
LEARNING_RATE = 0.1
TEST_RATIO = 0.25
RANDOM_SEED = 42

WEIGHTS_FILE = "weights.json"
BACKUP_DIR = "backups"

# =====================================================================
# NEURAL NETWORK ENGINE
# =====================================================================
class ConfigurableNN:
    def __init__(self, input_dim=INPUT_SIZE, hidden_layers=HIDDEN_LAYERS, output_dim=OUTPUT_SIZE, 
                 lr=LEARNING_RATE, weights_file=WEIGHTS_FILE, backup_dir=BACKUP_DIR):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.lr = lr
        self.weights_file = weights_file
        self.backup_dir = backup_dir
        
        self.weights, self.biases = self.load_or_create_weights()

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def _d_sigmoid(x):
        return x * (1 - x)

    def load_or_create_weights(self, filepath=None):
        target_file = filepath if filepath else self.weights_file
        if os.path.exists(target_file):
            with open(target_file, "r") as f:
                data = json.load(f)
            weights = [np.array(w) for w in data["weights"]]
            biases = [np.array(b) for b in data["biases"]]
            print(f"[+] Loaded weights from '{target_file}' (Saved: {data.get('timestamp', 'N/A')})")
            return weights, biases

        print(f"[!] '{target_file}' not found. Initializing random weights...")
        np.random.seed(RANDOM_SEED)
        layer_sizes = [self.input_dim] + self.hidden_layers + [self.output_dim]
        weights = [np.random.randn(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes) - 1)]
        biases = [np.random.randn(1, layer_sizes[i+1]) for i in range(len(layer_sizes) - 1)]
        
        self.save_state(weights, biases)
        return weights, biases

    def save_state(self, weights=None, biases=None):
        weights = weights if weights is not None else self.weights
        biases = biases if biases is not None else self.biases
        
        payload = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "weights": [w.tolist() for w in weights],
            "biases": [b.tolist() for b in biases]
        }
        
        with open(self.weights_file, "w") as f:
            json.dump(payload, f, indent=4)
            
        os.makedirs(self.backup_dir, exist_ok=True)
        backup_path = os.path.join(self.backup_dir, f"weights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(backup_path, "w") as f:
            json.dump(payload, f, indent=4)
            
        print(f"[+] Updated primary '{self.weights_file}' and backup '{backup_path}'")

    def forward(self, X):
        activations = [X]
        layer_input = X
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            layer_input = self._sigmoid(np.dot(layer_input, w) + b)
            activations.append(layer_input)
            
        output = self._sigmoid(np.dot(layer_input, self.weights[-1]) + self.biases[-1])
        activations.append(output)
        return activations

    def evaluate(self, X, Y):
        if len(X) == 0:
            return 0.0, 0.0
        preds = self.forward(X)[-1]
        mae = np.mean(np.abs(Y - preds))
        acc = np.mean(np.argmax(preds, axis=1) == np.argmax(Y, axis=1)) * 100.0
        return acc, mae

    def train(self, X_train, Y_train, X_test, Y_test, epochs=EPOCHS):
        print("\n--- Training Progress ---")
        for epoch in range(1, epochs + 1):
            activations = self.forward(X_train)
            
            error = Y_train - activations[-1]
            delta = error * self._d_sigmoid(activations[-1])
            deltas = [delta]
            
            for i in range(len(self.weights) - 2, -1, -1):
                delta = deltas[-1].dot(self.weights[i+1].T) * self._d_sigmoid(activations[i+1])
                deltas.append(delta)
            deltas.reverse()

            for i in range(len(self.weights)):
                self.weights[i] += self.lr * activations[i].T.dot(deltas[i])
                self.biases[i] += self.lr * np.sum(deltas[i], axis=0, keepdims=True)

            if epoch % (epochs // 5) == 0 or epoch == epochs:
                tr_acc, tr_mae = self.evaluate(X_train, Y_train)
                te_acc, te_mae = self.evaluate(X_test, Y_test)
                print(f"Epoch {epoch:5d}/{epochs} | Train MAE: {tr_mae:.5f} (Acc: {tr_acc:.1f}%) | Test MAE: {te_mae:.5f} (Acc: {te_acc:.1f}%)")

        self.save_state()

    def predict(self, X):
        activations = self.forward(X)
        return activations[-1]

# =====================================================================
# DATA PIPELINE (PARSING JSON / CSV DATASETS)
# =====================================================================
class DatasetLoader:
    def __init__(self):
        self.X = None
        self.Y = None
        self.min_vals = None
        self.max_vals = None
        self.class_map = ["Battery", "Solar Panel", "Grid"]

    def load_file(self, filepath):
        if not os.path.exists(filepath):
            print(f"[!] File not found: {filepath}")
            return False

        inputs, outputs = [], []
        
        # Load JSON format
        if filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                data = json.load(f)
            for sample in data:
                inputs.append([
                    sample["voltage"], sample["current"], sample["temperature"],
                    sample["humidity"], sample["wind_speed"]
                ])
                out = sample["power_source"]
                if out == "Battery": target = [1, 0, 0]
                elif out == "Solar Panel": target = [0, 1, 0]
                else: target = [0, 0, 1]
                outputs.append(target)
                
        # Load CSV format
        elif filepath.endswith('.csv'):
            raw = np.genfromtxt(filepath, delimiter=',', skip_header=1, dtype=str)
            for row in raw:
                inputs.append([float(x) for x in row[:5]])
                out = row[5].strip()
                if out == "Battery": target = [1, 0, 0]
                elif out == "Solar Panel": target = [0, 1, 0]
                else: target = [0, 0, 1]
                outputs.append(target)

        self.X = np.array(inputs)
        self.Y = np.array(outputs)
        
        # Min-Max Normalization
        self.min_vals = np.min(self.X, axis=0)
        self.max_vals = np.max(self.X, axis=0)
        denom = np.where((self.max_vals - self.min_vals) == 0, 1, self.max_vals - self.min_vals)
        self.X_norm = (self.X - self.min_vals) / denom
        
        print(f"[+] Loaded {len(self.X)} samples from '{filepath}'.")
        return True

    def get_split(self, test_ratio=TEST_RATIO, seed=RANDOM_SEED):
        if self.X is None:
            return None, None, None, None
        np.random.seed(seed)
        idx = np.arange(len(self.X_norm))
        np.random.shuffle(idx)
        split = int(len(self.X_norm) * (1 - test_ratio))
        return self.X_norm[idx[:split]], self.X_norm[idx[split:]], self.Y[idx[:split]], self.Y[idx[split:]]

    def normalize_sample(self, raw_sample):
        denom = np.where((self.max_vals - self.min_vals) == 0, 1, self.max_vals - self.min_vals)
        return (np.array(raw_sample) - self.min_vals) / denom

# =====================================================================
# INTERACTIVE CLI ENGINE
# =====================================================================
def run_cli():
    nn = ConfigurableNN()
    dataset = DatasetLoader()

    menu = """
===================================================
          NEURAL NETWORK MANAGEMENT CLI
===================================================
  [1] upload  - Load dataset (.json or .csv)
  [2] train   - Train neural network on dataset
  [3] test    - Test single input prediction
  [4] states  - View system architecture & stats
  [5] rollback- Restore weights from a backup file
  [6] exit    - Close application
==================================================="""

    print(menu)

    while True:
        cmd = input("\nNN-CLI> ").strip().lower()

        if cmd in ["1", "upload"]:
            filepath = input("Enter path to dataset file (.json/.csv): ").strip()
            dataset.load_file(filepath)

        elif cmd in ["2", "train"]:
            if dataset.X is None:
                print("[!] Upload a dataset first using option 1.")
                continue
            X_tr, X_te, Y_tr, Y_te = dataset.get_split()
            epochs_input = input(f"Enter epochs [{EPOCHS}]: ").strip()
            epochs = int(epochs_input) if epochs_input.isdigit() else EPOCHS
            nn.train(X_tr, Y_tr, X_te, Y_te, epochs=epochs)

        elif cmd in ["3", "test"]:
            if dataset.min_vals is None:
                print("[!] Load a dataset first to set feature normalization bounds.")
                continue
            try:
                print("Enter input values:")
                v = float(input("  Voltage: "))
                c = float(input("  Current: "))
                t = float(input("  Temperature: "))
                h = float(input("  Humidity: "))
                w = float(input("  Wind Speed: "))
                
                norm_inp = dataset.normalize_sample([v, c, t, h, w])
                raw_pred = nn.predict(np.array([norm_inp]))[0]
                predicted_class = dataset.class_map[np.argmax(raw_pred)]
                
                print(f"\n[RESULTS] Raw Output Probabilities: {np.round(raw_pred, 4)}")
                print(f"[RESULTS] Predicted Power Source: >> {predicted_class} <<")
            except ValueError:
                print("[!] Invalid numerical input.")

        elif cmd in ["4", "states"]:
            print("\n--- SYSTEM STATES ---")
            print(f"Network Structure : {INPUT_SIZE} Inputs -> {HIDDEN_LAYERS} Hidden -> {OUTPUT_SIZE} Outputs")
            print(f"Learning Rate     : {nn.lr}")
            print(f"Active Weights    : '{nn.weights_file}'")
            if dataset.X is not None:
                X_tr, X_te, Y_tr, Y_te = dataset.get_split()
                tr_acc, tr_mae = nn.evaluate(X_tr, Y_tr)
                te_acc, te_mae = nn.evaluate(X_te, Y_te)
                print(f"Dataset Size      : {len(dataset.X)} samples")
                print(f"Current Train Acc : {tr_acc:.2f}% (MAE: {tr_mae:.5f})")
                print(f"Current Test Acc  : {te_acc:.2f}% (MAE: {te_mae:.5f})")
            else:
                print("Dataset Status    : No dataset currently uploaded.")

        elif cmd in ["5", "rollback"]:
            backups = sorted(glob.glob(os.path.join(BACKUP_DIR, "*.json")), reverse=True)
            if not backups:
                print("[!] No backup files found in 'backups/' directory.")
                continue
            
            print("\nAvailable Backups:")
            for idx, b_file in enumerate(backups):
                print(f"  [{idx + 1}] {os.path.basename(b_file)}")
                
            choice = input(f"Select backup to restore (1-{len(backups)}): ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(backups):
                selected_backup = backups[int(choice) - 1]
                nn.weights, nn.biases = nn.load_or_create_weights(filepath=selected_backup)
                nn.save_state()
                print(f"[+] Successfully rolled back active state to '{selected_backup}'.")
            else:
                print("[!] Invalid choice.")

        elif cmd in ["6", "exit"]:
            print("Exiting Neural Network CLI.")
            break

        else:
            print("Unknown command. Type '1', '2', '3', '4', '5', or '6'.")

if __name__ == '__main__':
    run_cli()
