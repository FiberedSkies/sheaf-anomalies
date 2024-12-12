import random
import unswprocessing as unsw
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

class RestrictionMap(nn.Module):
    def __init__(self, source_dim, dest_dim, edge_dim, hidden_dim=40):
        super().__init__()
        self.source_mlp = nn.Sequential(
            nn.Linear(source_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(hidden_dim, edge_dim)
        )
        
        self.dest_mlp = nn.Sequential(
            nn.Linear(dest_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(hidden_dim, edge_dim)
        )
        self._init_weights()
        self.to(device)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, source_feat, dest_feat):
        if torch.isnan(source_feat).any() or torch.isnan(dest_feat).any():
            raise ValueError("Input features contain NaN values")
            
        if len(source_feat.shape) == 1:
            source_feat = source_feat.unsqueeze(0)
        if len(dest_feat.shape) == 1:
            dest_feat = dest_feat.unsqueeze(0)
            
        source_feat = source_feat.to(device)
        dest_feat = dest_feat.to(device)

        with torch.cuda.amp.autocast(enabled=True):
            source_restriction = self.source_mlp(source_feat)
            dest_restriction = self.dest_mlp(dest_feat)

        if torch.isnan(source_restriction).any() or torch.isnan(dest_restriction).any():
            print("Warning: NaN values detected in restrictions")
            source_restriction = torch.nan_to_num(source_restriction, 0.0)
            dest_restriction = torch.nan_to_num(dest_restriction, 0.0)

        return source_restriction, dest_restriction
    
class MetaLearner:
    def __init__(self, train, alpha=0.01, beta=0.005, gamma=0.001):
        self.trainingset = train
        self.mr1 = alpha
        self.mr2 = beta
        self.ftr = gamma

        self.metamap = RestrictionMap(
            source_dim=len(unsw.sfeat),
            dest_dim=len(unsw.dfeat),
            edge_dim=len(unsw.efeat)
        )
        self.optim = optim.Adam(self.metamap.parameters(), lr=self.mr2)

        self.emaps = {}
        self.eoptim = {}

    def coboundary_loss(self, srcrest, destrest, evec):
        srcrest = srcrest.to(device)
        destrest = destrest.to(device)
        evec = evec.to(device)
        
        combined = 2 * evec - srcrest - destrest
        norm = torch.norm(combined, p=2)

        return norm
    
    def metalearn(self, epochs=100, batch=32, msplit=0.25):
        print("[*] Starting metalearning phase...")
        metalearn = {}
        for edge, features in self.trainingset.items():

            if edge not in metalearn:
                metalearn[edge] = {"sfeat": [], "dfeat": [], "efeat": []}

            erecords = len(features["sfeat"])
            idx = sorted(random.sample(range(erecords), round(erecords * msplit)))

            for sections in ["sfeat", "dfeat", "efeat"]:
                metalearn[edge][sections] = [features[sections][i] for i in idx]
                self.trainingset[edge][sections] = [features[sections][i] for i in range(len(features[sections])) if i not in idx]
        
        self.metamap.train()
        total_losses = []
        for epoch in range(epochs):
            epoch_loss = 0
            meta_grads = {name: torch.zeros_like(param) for name, param in self.metamap.named_parameters()}
            n_edges = len(metalearn)
            for edge, features in metalearn.items():
                orig_params = {name: param.clone() for name, param in self.metamap.named_parameters()}
                edge_loss = 0
                n_batches = 0
                idcs = list(range(len(features["sfeat"])))
                random.shuffle(idcs)

                for batchidx in range(0, len(features["sfeat"]), batch):
                    batchidcs = idcs[batchidx:min(batchidx+batch, len(features["sfeat"]))]

                    batchsfeat = torch.tensor([features["sfeat"][j] for j in batchidcs]).to(device)
                    batchdfeat = torch.tensor([features["dfeat"][j] for j in batchidcs]).to(device)
                    batchefeat = torch.tensor([features["efeat"][j] for j in batchidcs]).to(device)

                    self.optim.zero_grad()

                    srcrest, destrest = self.metamap(batchsfeat, batchdfeat)
                    loss = self.coboundary_loss(srcrest, destrest, batchefeat)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.metamap.parameters(), max_norm=1.0)

                    with torch.no_grad():
                        for name, param in self.metamap.named_parameters():
                            param.data = param.data - self.mr1 * param.grad.data
                    
                    edge_loss += loss.item()
                    n_batches += 1
                
                epoch_loss += edge_loss / n_batches
                for name, param in self.metamap.named_parameters():
                    meta_grads[name] += param.grad.data / n_edges
            
                with torch.no_grad():
                    for name, param in self.metamap.named_parameters():
                        param.data.copy_(orig_params[name])
                
            with torch.no_grad():
                for name, param in self.metamap.named_parameters():
                    param.data -= self.mr2 * meta_grads[name] / len(metalearn)
            
            avg_epoch_loss = epoch_loss / n_edges
            total_losses.append(avg_epoch_loss)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Average Loss: {avg_epoch_loss:.4f}")

        self.metamap.eval()

    def initedges(self):
        print("[*] Initialize restriction maps over connections...")
        for edge, features in self.trainingset.items():
            self.emaps[edge] = RestrictionMap(
                source_dim=len(unsw.sfeat),
                dest_dim=len(unsw.dfeat),
                edge_dim=len(unsw.efeat)
            ).to(device)

            with torch.no_grad():
                for target_param, source_param in zip(self.emaps[edge].parameters(), self.metamap.parameters()):
                    target_param.data.copy_(source_param.data)
            
            self.eoptim[edge] = optim.Adam(self.emaps[edge].parameters(), lr=self.ftr)
    
    def finetune(self, epochs=100, batch=32):
        print("[*] Starting connection finetuning...")
        terminal_stats = {}
        for edge, emap in self.emaps.items():
            emap.train()
            features = self.trainingset[edge]

            final_losses = []

            for epoch in range(epochs):
                edge_loss = 0
                idcs = list(range(len(features["sfeat"])))
                random.shuffle(idcs)
                n_batches = 0
                for batchidx in range(0, len(features["sfeat"]), batch):
                    batchidcs = idcs[batchidx:min(batchidx+batch, len(features["sfeat"]))]

                    batchsfeat = torch.tensor([features["sfeat"][j] for j in batchidcs]).to(device)
                    batchdfeat = torch.tensor([features["dfeat"][j] for j in batchidcs]).to(device)
                    batchefeat = torch.tensor([features["efeat"][j] for j in batchidcs]).to(device)

                    self.eoptim[edge].zero_grad()
                    srcrest, destrest = emap(batchsfeat, batchdfeat)
                    loss = self.coboundary_loss(srcrest, destrest, batchefeat)

                    if epoch == epochs - 1:
                        final_losses.append(loss.item())

                    loss.backward()
                    self.eoptim[edge].step()

                    edge_loss += loss.item()
                    n_batches += 1

                if epoch % 10 == 0:
                    avg_loss = edge_loss / ((len(features["sfeat"]) + batch - 1) // batch)
                    print(f"Edge {edge}, Epoch {epoch}, Average Loss: {avg_loss:.4f}")
            mean_loss = np.mean(final_losses)
            std_loss = np.std(final_losses[:100])
            terminal_stats[edge] = {'mean': mean_loss, 'std': std_loss}
            print(f"Edge {edge} - Mean: {mean_loss:.4f}, Std: {std_loss:.4f}")

            emap.eval()
        print("[+] Training complete!")
        return terminal_stats

class Detector(MetaLearner):
    def __init__(self, train, tests, alpha=0.15, beta=0.1, gamma=0.005):
        super().__init__(train, alpha, beta, gamma)
        self.thresholds = {}
        self.tests = tests
    
    def counts(self):
        counts = {}
        for ar in self.tests.keys():
            counts[ar] = self.tests[ar]['label'].count(1)
        return counts

    def train(self, epochs=350, batch=6, msplit=0.20):
        self.metalearn(epochs, batch, msplit)
        self.initedges()
        stats = self.finetune(epochs, batch)
        for edge, stat in stats.items():
            self.thresholds[edge] = stat['mean'] + 0.25 * stat['std']
        
    def detect(self):
        print("[*] Starting detection validation...")
        ar_results = {}
        ar_metrics = {}
        for ar, testset in self.tests.items():
            results = {}
            metrics = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0, 'attack_types': {}}
            for edge, features in testset.items():
                if edge not in self.emaps:
                    continue

                edge_results = {'predictions': [], 'metrics': {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}}

                sfeat = torch.tensor(features["sfeat"]).to(device)
                dfeat = torch.tensor(features["dfeat"]).to(device)
                efeat = torch.tensor(features["efeat"]).to(device)
                predictions = []
                self.emaps[edge].eval()
                with torch.no_grad():
                    for i in range(len(sfeat)):
                        srcrest, destrest = self.emaps[edge](sfeat[i], dfeat[i])
                        loss = self.coboundary_loss(srcrest, destrest, efeat[i])
                        predictions.append(1 if loss > self.thresholds[edge] else 0)

                    for i, (pred, tlabel, attack) in enumerate(zip(predictions, features['label'], features['attack'])):
                        edge_results['predictions'].append(pred)

                        if pred == 1 and tlabel == 1:
                            edge_results['metrics']['tp'] += 1
                            metrics['tp'] += 1

                            if attack not in metrics['attack_types']:
                                metrics['attack_types'][attack] = 0
                            metrics['attack_types'][attack] += 1
                        elif pred == 1 and tlabel == 0:
                            edge_results['metrics']['fp'] += 1
                            metrics['fp'] += 1
                        elif pred == 0 and tlabel == 0:
                            edge_results['metrics']['tn'] += 1
                            metrics['tn'] += 1
                        elif pred == 0 and tlabel == 1:
                            edge_results['metrics']['fn'] += 1
                            metrics['fn'] += 1
                
                results[edge] = edge_results

            ar_results[ar] = results
            ar_metrics[ar] = metrics
        return ar_results, ar_metrics
    
    def analyze_metrics(self, ar_results, ar_metrics):
        for ar, metrics in ar_metrics.items():
            print(f"\nResults for {ar*100}% anomaly rate:")

            total_pred = metrics['tp'] + metrics['fp'] + metrics['tn'] + metrics['fn']
            precision = metrics['tp'] / (metrics['tp'] + metrics['fp']) if (metrics['tp'] + metrics['fp']) > 0 else 0
            recall = metrics['tp'] / (metrics['tp'] + metrics['fn']) if (metrics['tp'] + metrics['fn']) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (metrics['tp'] + metrics['tn']) / total_pred if total_pred > 0 else 0

            print(f"Overall Metrics:")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"true anomalies: {metrics['tp']}")
            print(f"false anomalies: {metrics['fp']}")
            print(f"true benign: {metrics['tn']}")
            print(f"false benign: {metrics['fn']}")

            print("\nAttack Type Analysis:")

            total_attack_counts = {}
            for edge, features in self.tests[ar].items():
                for attack in features['attack']:
                    if attack not in total_attack_counts:
                        total_attack_counts[attack] = 0
                    total_attack_counts[attack] += 1

            print(f"{'Attack Type':<20} {'Total':<8} {'Detected':<10} {'Detection Rate':<15}")
            print("-" * 55)
            
            for attack_type in total_attack_counts.keys():
                total = total_attack_counts[attack_type]
                detected = metrics['attack_types'].get(attack_type, 0)
                detection_rate = (detected / total * 100) if total > 0 else 0
                print(f"{attack_type:<20} {total:<8} {detected:<10} {detection_rate:>6.1f}%")

anoms = [0.05, 0.1, 0.2]
train, tests = unsw.process(anoms)

print(f"Training set size {unsw.record_count(train)}")
for ar in anoms:
    print(f"[+] For anomaly rate {ar*100}%, test set size {unsw.record_count(tests[ar])}")

detect = Detector(train, tests)
detect.train()
results, metrics = detect.detect()
detect.analyze_metrics(results, metrics)