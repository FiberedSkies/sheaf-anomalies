import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict, Counter
import sheaf_preprocessing as pre
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RestrictionMap(nn.Module):
    def __init__(self, source_dim, dest_dim, edge_dim, hidden_dim=24):
        super().__init__()
        self.source_mlp = nn.Sequential(
            nn.Linear(source_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 14)
        )
        
        self.dest_mlp = nn.Sequential(
            nn.Linear(dest_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 14)
        )
        self.to(device)

    def forward(self, source_feat, dest_feat):
        source_restriction = self.source_mlp(source_feat)
        dest_restriction = self.dest_mlp(dest_feat)
        return source_restriction, dest_restriction

class MetaLearner:
    def __init__(self, trainset, alpha=0.005, beta=0.005, gamma=0.001):
        self.trainset = trainset
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.metamap = RestrictionMap(
            source_dim=len(trainset["sfeats"][0][0]),
            dest_dim=len(trainset["dfeats"][0][0]),
            edge_dim=len(trainset["efeats"][0][0])
        ).to(device)
        self.base_optimizer = optim.Adam(self.metamap.parameters(), lr=self.beta)

        self.edge_maps = {}
        self.edge_optimizers = {}
        
    def coboundary(self, src_rest, dst_rest, edge_feat):
        """Compute δ² as defined in Definition 2.3 of the paper"""
        return torch.norm(src_rest + dst_rest - 2 * edge_feat, p=2)
        
    def train_metamap(self, num_epochs=85, batch_size=32, msr=0.2):
        """Train the base restriction map on a subset of the data"""
        counts = Counter(self.trainset["enums"][0])
        total = len(self.trainset["enums"][0])
        edge_distribution = {enum: count/total for enum, count in counts.items()}

        all_edges = list(zip(
                self.trainset["srcs"][0],
                self.trainset["dsts"][0],
                self.trainset["sfeats"][0],
                self.trainset["dfeats"][0],
                self.trainset["efeats"][0],
                self.trainset["enums"][0]
            ))

        sampling_probabilities = [edge_distribution[edge[5]] for edge in all_edges]
        sampling_probabilities = np.array(sampling_probabilities)
        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        meta_sample_size = int(len(all_edges) * msr)
        self.metamap.train()
        used_indices = set()

        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0
            
            msi = np.random.choice(
                len(all_edges),
                size=meta_sample_size,
                p=sampling_probabilities,
                replace=False
            )
            medges = [all_edges[i] for i in msi]
            used_indices.update(edge[5] for edge in medges)

            for i in range(0, len(medges), batch_size):

                batch = medges[i:i + batch_size]
                batch_loss = 0

                for src, dst, src_feat, dst_feat, edge_feat, enum in batch:
                    src_feat = torch.FloatTensor(src_feat).to(device)
                    dst_feat = torch.FloatTensor(dst_feat).to(device)
                    edge_feat = torch.FloatTensor(edge_feat).to(device)

                    self.base_optimizer.zero_grad()
                    src_rest, dst_rest = self.metamap(src_feat, dst_feat)
                    loss = self.coboundary(src_rest, dst_rest, edge_feat)
                    loss.backward()
                    self.base_optimizer.step()

                    batch_loss += loss.cpu().item()

                total_loss += batch_loss
                num_batches += 1
            
            if epoch % 5 == 0:
                avg_loss = total_loss / max(num_batches, 1)
                print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")

        enum_to_idx = {enum: idx for idx, enum in enumerate(self.trainset["enums"][0])}

        used_idx = [enum_to_idx[edge_id] for edge_id in used_indices]
        
        mask = np.ones(len(self.trainset["enums"][0]), dtype=bool)
        mask[used_idx] = False

        self.trainset["srcs"][0] = [x for i, x in enumerate(self.trainset["srcs"][0]) if mask[i]]
        self.trainset["dsts"][0] = [x for i, x in enumerate(self.trainset["dsts"][0]) if mask[i]]
        self.trainset["sfeats"][0] = [x for i, x in enumerate(self.trainset["sfeats"][0]) if mask[i]]
        self.trainset["dfeats"][0] = [x for i, x in enumerate(self.trainset["dfeats"][0]) if mask[i]]
        self.trainset["efeats"][0] = [x for i, x in enumerate(self.trainset["efeats"][0]) if mask[i]]
        self.trainset["enums"][0] = [x for i, x in enumerate(self.trainset["enums"][0]) if mask[i]]

        self.metamap.eval()
    
    def init_edgemaps(self):
        """Initialize edge-specific maps using the trained base map"""
        unique_edges = set(self.trainset["enums"][0])
        
        for edge in unique_edges:
            self.edge_maps[edge] = RestrictionMap(
                source_dim=len(self.trainset["sfeats"][0][0]),
                dest_dim=len(self.trainset["dfeats"][0][0]),
                edge_dim=len(self.trainset["efeats"][0][0])
            ).to(device)
            
            for target_param, source_param in zip(
                self.edge_maps[edge].parameters(), 
                self.metamap.parameters()
            ):
                target_param.data.copy_(source_param.data)
            
            self.edge_optimizers[edge] = optim.Adam(
                self.edge_maps[edge].parameters(),
                lr=self.gamma
            )

    def finetune_single_edge(self, edge, edge_data, num_epochs=100):
        """Train a single edge map and monitor its behavior closely."""
        edge_map = self.edge_maps[edge]
        optimizer = self.edge_optimizers[edge]
        
        # Convert all data for this edge upfront
        src_feats = torch.FloatTensor([d['src_feat'] for d in edge_data]).to(device)
        dst_feats = torch.FloatTensor([d['dst_feat'] for d in edge_data]).to(device)
        edge_feats = torch.FloatTensor([d['edge_feat'] for d in edge_data]).to(device)
        
        # Check initial loss
        with torch.no_grad():
            src_rest, dst_rest = edge_map(src_feats, dst_feats)
            initial_loss = self.coboundary(src_rest, dst_rest, edge_feats).item()
            print(f"Edge {edge} initial loss: {initial_loss}")

        losses = []
        best_loss = float('inf')
        patience = 25
        patience_counter = 0
        
        for epoch in range(num_epochs):
            edge_map.train()
            optimizer.zero_grad()
            
            src_rest, dst_rest = edge_map(src_feats, dst_feats)
            loss = self.coboundary(src_rest, dst_rest, edge_feats)
            
            loss.backward()
            optimizer.step()
            
            current_loss = loss.item()
            losses.append(current_loss)
            
            if current_loss < best_loss:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping for edge {edge} at epoch {epoch}")
                break
                
            if epoch % 10 == 0:
                print(f"Edge {edge} - Epoch {epoch}, Loss: {current_loss:.4f}")
                
            if current_loss > initial_loss * 2:
                print(f"Warning: Loss increased significantly for edge {edge}")
                break
        
        return best_loss

    def finetune(self, num_epochs=100, batch_size=32):
        """Train edge-specific maps sequentially."""
        edge_data = defaultdict(list)
        for idx in range(len(self.trainset["enums"][0])):
            edge = self.trainset["enums"][0][idx]
            edge_data[edge].append({
                'src_feat': self.trainset["sfeats"][0][idx],
                'dst_feat': self.trainset["dfeats"][0][idx],
                'edge_feat': self.trainset["efeats"][0][idx]
            })
        
        final_losses = {}
        for edge in edge_data.keys():
            n_samples = len(edge_data[edge])
            adjusted_lr = self.gamma * min(1.0, n_samples/100)
            self.edge_optimizers[edge] = optim.Adam(
                self.edge_maps[edge].parameters(),
                lr=adjusted_lr
            )
            print(f"\nFine-tuning edge: {edge}")
            print(f"Number of samples: {len(edge_data[edge])}")
            final_losses[edge] = self.finetune_single_edge(edge, edge_data[edge], num_epochs)
        
        return final_losses

    def metalearn(self, num_epochs=100, batch_size=32):
        """full learning process"""

        self.train_metamap()
        self.init_edgemaps()
        final_losses = self.finetune()

        return self.edge_maps, final_losses
    
class AnomalyClassifier(MetaLearner):
    def __init__(self, trainset, testsets, alpha=0.005, beta=0.001, gamma=0.001):
        super().__init__(trainset, alpha, beta, gamma)
        self.testsets = testsets
        self.thresholds = None

    def compute_attack_metrics(self, predictions, true_labels, attack_types):
        """
        Compute detection metrics for each attack type.
        """
        attack_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'total': 0})
        
        for pred, true_label, attack in zip(predictions, true_labels, attack_types):
            if true_label == 1:
                attack_metrics[attack]['total'] += 1
                if pred == 1:
                    attack_metrics[attack]['tp'] += 1
                else:
                    attack_metrics[attack]['fn'] += 1
            elif pred == 1:
                attack_metrics[attack]['fp'] += 1
                
        attack_performance = {}
        for attack, metrics in attack_metrics.items():
            if metrics['total'] > 0:
                precision = metrics['tp'] / (metrics['tp'] + metrics['fp']) if (metrics['tp'] + metrics['fp']) > 0 else 0
                recall = metrics['tp'] / (metrics['tp'] + metrics['fn']) if metrics['total'] > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                attack_performance[attack] = {
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'total_occurrences': metrics['total'],
                    'detected': metrics['tp'],
                    'missed': metrics['fn']
                }
        
        return attack_performance
    
    def classify(self, thresholds):
        """
        Classify anomalies in test sets using learned edge maps and thresholds.
        """
        results = {}
        
        for anomaly_rate, testset in self.testsets.items():
            print(f"\nEvaluating test set with {anomaly_rate*100}% anomaly rate")
            
            predictions = []
            true_labels = testset["labels"][0]
            attack_types = testset["attack_type"][0]
            
            test_edges = list(zip(
                testset["srcs"][0],
                testset["dsts"][0],
                testset["sfeats"][0],
                testset["dfeats"][0],
                testset["efeats"][0]
            ))
            
            for src, dst, src_feat, dst_feat, edge_feat in test_edges:
                enum = src + " to " + dst
                
                if enum not in self.edge_maps:
                    predictions.append(1)
                    continue
                
                src_tensor = torch.FloatTensor(src_feat).to(device)
                dst_tensor = torch.FloatTensor(dst_feat).to(device)
                edge_tensor = torch.FloatTensor(edge_feat).to(device)
                
                edge_map = self.edge_maps[enum]
                with torch.no_grad():
                    src_rest, dst_rest = edge_map(src_tensor, dst_tensor)
                    cob_score = self.coboundary(src_rest, dst_rest, edge_tensor).cpu().item()
                
                predictions.append(int(cob_score > thresholds[enum]))
            
            predictions = np.array(predictions)
            true_labels = np.array(true_labels)
            
            tp = np.sum((predictions == 1) & (true_labels == 1))
            fp = np.sum((predictions == 1) & (true_labels == 0))
            tn = np.sum((predictions == 0) & (true_labels == 0))
            fn = np.sum((predictions == 0) & (true_labels == 1))
            
            total = len(true_labels)
            accuracy = (tp + tn) / total
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            attack_performance = self.compute_attack_metrics(predictions, true_labels, attack_types)
            
            results[anomaly_rate] = {
                'overall_metrics': {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'confusion_matrix': {
                        'tp': int(tp),
                        'fp': int(fp),
                        'tn': int(tn),
                        'fn': int(fn)
                    }
                },
                'attack_type_metrics': attack_performance,
                'predictions': predictions.tolist(),
                'true_labels': true_labels.tolist(),
                'attack_types': attack_types
            }
            
            print(f"\nOverall Results for {anomaly_rate*100}% anomaly rate:")
            print(f"Total samples: {total}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            
            print("\nPerformance by Attack Type:")
            for attack, metrics in attack_performance.items():
                print(f"\n{attack}:")
                print(f"Total occurrences: {metrics['total_occurrences']}")
                print(f"Successfully detected: {metrics['detected']}")
                print(f"Missed detections: {metrics['missed']}")
                print(f"Precision: {metrics['precision']:.4f}")
                print(f"Recall: {metrics['recall']:.4f}")
                print(f"F1 Score: {metrics['f1_score']:.4f}")
        
        return results
    
    def fullrun(self):
        print("Starting anomaly detection pipeline...")
        print("\nPhase 1: Meta-learning and edge map training")
        edge_maps, thresholds = self.metalearn()
        print("\nPhase 2: Anomaly classification")        
        results = self.classify(thresholds)
        print("\nPipeline completed successfully.")
        return results
    
trainset, testsets = pre.preprocess(300000, 0.15, 9, [0.05, 0.1, 0.2])

edges_before = Counter(trainset["enums"][0])
for ar, testset in testsets.items():
    print(f"\nAnalyzing test set with {ar*100}% anomaly rate:")
    
    test_edges = set()
    anomaly_edges = set()
    for src, dst, label in zip(
        testset["srcs"][0],
        testset["dsts"][0],
        testset["labels"][0]
    ):
        edge = f"{src} to {dst}"
        test_edges.add(edge)
        if label == 1:
            anomaly_edges.add(edge)
    
    print(f"Total unique test edges: {len(test_edges)}")
    print(f"Unique anomaly edges: {len(anomaly_edges)}")
    
    covered_anomalies = anomaly_edges.intersection(edges_before.keys())
    print(f"Covered anomaly edges: {len(covered_anomalies)}")
    
    print("\nTraining samples for anomaly edges:")
    for edge in covered_anomalies:
        print(f"Edge {edge}: {edges_before[edge]} training samples")

print(f"Number of unique edges: {len(edges_before)}")
print(f"Total samples: {len(trainset['enums'][0])}")
print(f"Sample distribution:")
for edge, count in edges_before.most_common(150):
    print(f"Edge {edge}: {count} samples")

classifier = AnomalyClassifier(trainset, testsets)
classifier.fullrun()
