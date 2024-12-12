import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
import matplotlib.pyplot as plt
from unswprocessing import process, sfeat, dfeat, efeat

class NetworkAutoencoder(nn.Module):
    def __init__(self):
        super(NetworkAutoencoder, self).__init__()
        self.sfeat_dim = len(sfeat)  # 15
        self.dfeat_dim = len(dfeat)  # 15
        self.efeat_dim = len(efeat)  # 14
        self.input_dim = self.sfeat_dim + self.dfeat_dim + self.efeat_dim  # 44

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4)
        )

        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, self.input_dim),
            nn.Tanh()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    
    def get_reconstruction_error(self, x):
        """Calculate reconstruction error for anomaly detection"""
        encoded, decoded = self(x)
        return torch.mean(torch.square(x - decoded), dim=1)

class NetworkAnalyzer:
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = NetworkAutoencoder().to(self.device)
        
    def prepare_batch_data(self, edge_dict):
        """Convert edge dictionary data into batched tensor format."""
        features = []
        for edge in edge_dict:
            edge_data = edge_dict[edge]
            sfeat = edge_data['sfeat']
            dfeat = edge_data['dfeat']
            efeat = edge_data['efeat']
            
            for i in range(len(sfeat)):
                combined = np.concatenate([sfeat[i], dfeat[i], efeat[i]])
                features.append(combined)
        
        return np.array(features)

    def prepare_test_data(self, test_dict):
        """Prepare test data and labels for each anomaly rate."""
        test_data = {}
        test_labels = {}
        
        for ar in test_dict:
            ar_features = []
            ar_labels = []
            
            for edge in test_dict[ar]:
                edge_data = test_dict[ar][edge]
                sfeat = edge_data['sfeat']
                dfeat = edge_data['dfeat']
                efeat = edge_data['efeat']
                labels = edge_data['label']
                
                for i in range(len(sfeat)):
                    combined = np.concatenate([sfeat[i], dfeat[i], efeat[i]])
                    ar_features.append(combined)
                    ar_labels.append(labels[i])
            
            test_data[ar] = np.array(ar_features)
            test_labels[ar] = np.array(ar_labels)
        
        return test_data, test_labels

    def train_autoencoder(self, train_data, epochs=200, batch_size=32):
        """Train the autoencoder on the prepared data."""
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        train_tensor = torch.FloatTensor(train_data).to(self.device)
        train_dataset = torch.utils.data.TensorDataset(train_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True
        )
        
        losses = []
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                x = batch[0]
                _, decoded = self.model(x)
                loss = criterion(decoded, x)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}')

        self.model.eval()
        with torch.no_grad():
            errors = self.model.get_reconstruction_error(train_tensor)
            self.threshold = torch.quantile(errors, 0.95)
        
        return losses

    def detect_anomalies(self, data):
        """Detect anomalies using reconstruction error."""
        self.model.eval()
        with torch.no_grad():
            tensor_data = torch.FloatTensor(data).to(self.device)
            errors = self.model.get_reconstruction_error(tensor_data)
            predictions = (errors > self.threshold).cpu().numpy().astype(int)
            return predictions, errors.cpu().numpy()

    def evaluate(self, test_data, test_labels):
        """Evaluate anomaly detection performance."""
        results = {}
        for ar in test_data:
            predictions, errors = self.detect_anomalies(test_data[ar])
            
            # Calculate metrics
            from sklearn.metrics import precision_recall_fscore_support, accuracy_score
            precision, recall, f1, _ = precision_recall_fscore_support(test_labels[ar], predictions, average='binary')
            accuracy = accuracy_score(test_labels[ar], predictions)
            
            results[ar] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy,
                'predictions': predictions,
                'errors': errors
            }
        
        return results

    def run_analysis(self, train_dict, test_dict):
        """Run the complete analysis pipeline."""
        print("Preparing training data...")
        train_data = self.prepare_batch_data(train_dict)
        
        print("Preparing test data...")
        test_data, test_labels = self.prepare_test_data(test_dict)
        
        print("\nTraining autoencoder...")
        losses = self.train_autoencoder(train_data)
        
        print("\nPerforming anomaly detection evaluation...")
        results = self.evaluate(test_data, test_labels)
        
        return results, losses

def plot_losses(losses):
    """Plot training losses."""
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Autoencoder Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

def main():
    anomaly_rates = [0.05, 0.1, 0.2]

    print("Loading and preprocessing data...")
    train, tests = process(anomaly_rates)

    analyzer = NetworkAnalyzer()
    results, losses = analyzer.run_analysis(train, tests)
    
    print("\nResults:")
    for ar in results:
        print(f"\nAnomaly rate: {ar*100}%")
        print(f"Accuracy: {results[ar]['accuracy']:.4f}")
        print(f"Precision: {results[ar]['precision']:.4f}")
        print(f"Recall: {results[ar]['recall']:.4f}")
        print(f"F1 Score: {results[ar]['f1']:.4f}")

if __name__ == "__main__":
    main()