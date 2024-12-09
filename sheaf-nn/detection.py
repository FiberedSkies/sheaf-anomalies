import random
import unswprocessing as unsw
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

class RestrictionMap(nn.Module):
    def __init__(self, source_dim, dest_dim, edge_dim, hidden_dim=32):
        super().__init__()
        self.source_mlp = nn.Sequential(
            nn.Linear(source_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, edge_dim)
        )
        
        self.dest_mlp = nn.Sequential(
            nn.Linear(dest_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, edge_dim)
        )
        self.to(device)

    def forward(self, source_feat, dest_feat):
        source_feat = source_feat.to(device)
        dest_feat = dest_feat.to(device)

        print("Source feat stats:", 
          "min:", torch.min(source_feat).item(),
          "max:", torch.max(source_feat).item(),
          "mean:", torch.mean(source_feat).item())

        source_restriction = self.source_mlp(source_feat)
        print("Source restriction stats:", 
            "min:", torch.min(source_restriction).item(),
            "max:", torch.max(source_restriction).item(),
            "mean:", torch.mean(source_restriction).item())
        dest_restriction = self.dest_mlp(dest_feat)
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
        combined = srcrest + destrest - 2 * evec
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
        for epoch in range(epochs):
            meta_grads = {name: torch.zeros_like(param) for name, param in self.metamap.named_parameters()}
            for edge, features in metalearn.items():
                orig_params = {name: param.clone() for name, param in self.metamap.named_parameters()}
                edge_loss = 0
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
                for name, param in self.metamap.named_parameters():
                    meta_grads[name] += (param.data - orig_params[name]) / self.mr1
            
                with torch.no_grad():
                    for name, param in self.metamap.named_parameters():
                        param.data.copy_(orig_params[name])
                
            with torch.no_grad():
                for name, param in self.metamap.named_parameters():
                    param.data -= self.mr2 * meta_grads[name] / len(metalearn)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Average Loss: {edge_loss/len(metalearn):.4f}")

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
        terminal_losses = {}
        for edge, emap in self.emaps.items():
            emap.train()
            features = self.trainingset[edge]

            final_epoch_loss = 0

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

                    loss.backward()
                    self.eoptim[edge].step()

                    edge_loss += loss.item()
                    n_batches += 1

                if epoch == epochs - 1:
                    final_epoch_loss = edge_loss / n_batches
                if epoch % 10 == 0:
                    avg_loss = edge_loss / ((len(features["sfeat"]) + batch - 1) // batch)
                    print(f"Edge {edge}, Epoch {epoch}, Average Loss: {avg_loss:.4f}")
            
            terminal_losses[edge] = final_epoch_loss
            emap.eval()
        print("[+] Training complete!")
        return terminal_losses

class Detector(MetaLearner):
    def __init__(self, train, tests, alpha=0.01, beta=0.005, gamma=0.001):
        super().__init__(train, alpha, beta, gamma)
        self.thresholds = {}
        self.tests = tests

    def train(self, epochs=100, batch=32, msplit=0.25, epsilon=0.1):
        self.metalearn(epochs, batch, msplit)
        self.initedges()
        thresholds = self.finetune(epochs, batch)
        for edge, th in thresholds.items():
            self.thresholds[edge] = th + epsilon
    
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

                self.emaps[edge].eval()
                with torch.no_grad():
                    srcrest, destrest = self.emaps[edge](sfeat, dfeat)
                    losses = self.coboundary_loss(srcrest, destrest, efeat)

                    predictions = [1 if loss > self.thresholds[edge] else 0 for loss in losses]
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
            
            edge_metrics = []
            for edge, results in ar_results[ar].items():
                m = results['metrics']
                edge_total = sum(m.values())
                if edge_total > 0:
                    edge_precision = m['tp'] / (m['tp'] + m['fp']) if (m['tp'] + m['fp']) > 0 else 0
                    edge_recall = m['tp'] / (m['tp'] + m['fn']) if (m['tp'] + m['fn']) > 0 else 0
                    edge_f1 = 2 * (edge_precision * edge_recall) / (edge_precision + edge_recall) if (edge_precision + edge_recall) > 0 else 0
                    edge_accuracy = (m['tp'] + m['tn']) / edge_total
                    edge_metrics.append({
                        'edge': edge,
                        'precision': edge_precision,
                        'recall': edge_recall,
                        'f1': edge_f1,
                        'accuracy': edge_accuracy
                    })
            
            if edge_metrics:
                edge_f1s = [m['f1'] for m in edge_metrics]
                print(f"\nEdge Performance:")
                print(f"Average Edge F1: {np.mean(edge_f1s):.4f} ± {np.std(edge_f1s):.4f}")
                
                best_edge = max(edge_metrics, key=lambda x: x['f1'])
                worst_edge = min(edge_metrics, key=lambda x: x['f1'])
                print(f"Best Edge: {best_edge['edge']} (F1: {best_edge['f1']:.4f})")
                print(f"Worst Edge: {worst_edge['edge']} (F1: {worst_edge['f1']:.4f})")
            
            print("\nAttack Type Distribution:")
            total_detected = sum(metrics['attack_types'].values())
            if total_detected > 0:
                sorted_attacks = sorted(metrics['attack_types'].items(), key=lambda x: x[1], reverse=True)
                for attack_type, count in sorted_attacks:
                    percentage = (count / total_detected) * 100
                    print(f"{attack_type}: {count} ({percentage:.1f}%)")

anoms = [0.05, 0.1, 0.2]
train, tests = unsw.process(anoms)

print(f"Training set size {unsw.record_count(train)}")
for ar in anoms:
    print(f"[+] For anomaly rate {ar*100}%, test set size {unsw.record_count(tests[ar])}")

detect = Detector(train, tests)
detect.train()
results, metrics = detect.detect()
detect.analyze_metrics(results, metrics)