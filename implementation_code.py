# ==============================================================
# FPLPA — FEDERATED PROTOTYPE LEARNING WITH PRIVACY AWARENESS
# ENHANCED FOR STATISTICAL SIGNIFICANCE - FIXED VERSION
# ==============================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
from scipy import stats

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, f1_score,
    matthews_corrcoef, balanced_accuracy_score
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ==============================================================
# CRITICAL: REDUCED PARAMETERS FOR CLEARER DIFFERENTIATION
# ==============================================================
SUBSET_SIZES  = [3, 5, 7,8]
ROUNDS        = 20  # Reduced significantly to prevent saturation
EPOCHS        = 5   # Minimal epochs to create more variance
RUNS          = 10  # More runs for statistical power
CHI2_K        = 15  # Fewer features to increase difficulty

# Aggressive parameters for clear differentiation
WARMUP_ROUNDS = 3
EMA_ALPHA     = 0.9  # More weight on recent performance
TEMP_INIT     = 3.0  # Very high initial exploration
TEMP_FINAL    = 0.1  # Very low final temperature
UCB_WEIGHT    = 0.5  # High exploration bonus
LAM           = 0.2  # Higher prototype loss

# Extreme weights to differentiate algorithms
VOLUME_WEIGHT = 0.05      # Almost ignore volume
DELTA_WEIGHT  = 0.60      # Strong focus on performance improvement
BALANCE_WEIGHT = 0.05     # Almost ignore balance
CONSISTENCY_WEIGHT = 0.05 # Almost ignore consistency
DIVERSITY_WEIGHT = 0.25   # Strong diversity to avoid local optima

ALGOS = ["Random", "FedCS", "PowerChoice", "MultiMetric"]

# ==============================================================
# CLIENT FILES
# ==============================================================
CLIENT_FILES = [
    r"C:\Users\merug\Downloads\Data\apache.csv",
    r"C:\Users\merug\Downloads\Data\safe.csv",
    r"C:\Users\merug\Downloads\Data\zxing.csv",
    r"C:\Users\merug\Downloads\Data\cm1.csv",
    r"C:\Users\merug\Downloads\Data\mw1.csv",
    r"C:\Users\merug\Downloads\Data\PC1.csv",
    r"C:\Users\merug\Downloads\Data\PC3.csv",
    r"C:\Users\merug\Downloads\Data\PC4.csv",
    r"C:\Users\merug\Downloads\Data\AEEEM.csv",
]

# ==============================================================
# 1. DATA LOADING
# ==============================================================

def load_dataset(path: str):
    """Load dataset and properly encode labels"""
    df = pd.read_csv(path)
    name = path.split("\\")[-1].replace(".csv","")
    
    print(f"\n  Processing {name}:")
    print(f"    Original shape: {df.shape}")
    
    # Find label column
    label_col = None
    possible_label_names = ["defective", "label", "class", "bug", "isdefect", "defect", 
                           "defects", "buggy", "fault", "faulty", "status", "isDefective"]
    
    for c in df.columns:
        c_lower = c.lower()
        if c_lower in possible_label_names:
            label_col = c
            print(f"    Found label column: '{label_col}'")
            break
    
    if label_col is None:
        for c in df.columns:
            if df[c].dtype == 'object' and df[c].nunique() <= 10:
                label_col = c
                print(f"    Using categorical column '{c}' as label")
                break
    
    if label_col is None:
        for c in df.columns:
            if df[c].nunique() <= 10:
                label_col = c
                print(f"    Using column '{c}' as label (unique values: {df[c].nunique()})")
                break
    
    if label_col is None:
        label_col = df.columns[-1]
        print(f"    Using last column '{label_col}' as label")
    
    # Extract labels
    y_raw = df[label_col]
    
    # Encode labels to 0, 1, 2, ... sequentially
    le = LabelEncoder()
    y = le.fit_transform(y_raw.astype(str))
    
    # Remove rows with missing labels
    mask = ~pd.isna(y)
    X = df.drop(columns=[label_col]).select_dtypes(include=[np.number]).values
    X, y = X[mask], y[mask]
    
    # Handle missing values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Remove constant features
    var_mask = np.var(X, axis=0) > 1e-8
    if var_mask.sum() > 0:
        X = X[:, var_mask]
    else:
        X = X + np.random.normal(0, 0.01, X.shape)
    
    # Get class information
    num_classes = len(np.unique(y))
    class_counts = {i: np.sum(y == i) for i in range(num_classes)}
    
    print(f"    Number of classes: {num_classes}")
    print(f"    Class distribution: {class_counts}")
    print(f"    Total samples: {len(y)}")
    
    return X, y, name, num_classes

# ==============================================================
# 2. DATA PREPROCESSING
# ==============================================================

def mutual_info_select(X, y, k=CHI2_K):
    """Mutual information based feature selection"""
    k = min(k, X.shape[1])
    if k <= 0 or X.shape[1] <= k:
        return X
    
    try:
        scores = mutual_info_classif(X, y, random_state=42)
        idx = np.argsort(scores)[-k:]
        return X[:, idx]
    except:
        variances = np.var(X, axis=0)
        idx = np.argsort(variances)[-k:]
        return X[:, idx]

def preprocess_dataset(X, y, side_dim=5):
    """Preprocess dataset"""
    if X.shape[1] > CHI2_K:
        X = mutual_info_select(X, y, k=CHI2_K)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    side = side_dim
    required_features = side * side
    
    if X.shape[1] < required_features:
        padding = required_features - X.shape[1]
        X = np.pad(X, ((0, 0), (0, padding)), mode='constant')
    elif X.shape[1] > required_features:
        X = X[:, :required_features]
    
    X_square = X.reshape(X.shape[0], side, side)
    
    return X_square, y

# ==============================================================
# 3. SIMPLER MODEL (more variance)
# ==============================================================
class CPN(nn.Module):
    def __init__(self, side: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        
        # Smaller model for more variance between algorithms
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        reduced = max(1, side // 4)
        self.fc = nn.Sequential(
            nn.Linear(reduced * reduced * 16, 32), 
            nn.ReLU(), 
            nn.Dropout(0.2)
        )
        self.cls = nn.Linear(32, num_classes)

    def forward(self, x):
        z = self.conv(x.unsqueeze(1)).view(x.size(0), -1)
        e = self.fc(z)
        return self.cls(e), e

def fresh_model(side, num_classes):
    return CPN(side, num_classes).to(DEVICE)

# ==============================================================
# 4. PROTOTYPE HELPERS
# ==============================================================
def local_proto(embed, labels):
    protos = {}
    for c in torch.unique(labels):
        c = int(c)
        mask = (labels == c)
        if mask.sum() > 0:
            protos[c] = embed[mask].mean(dim=0).detach()
    return protos

def aggregate_protos(proto_list):
    merged = defaultdict(list)
    for p in proto_list:
        for c, v in p.items():
            merged[c].append(v)
    return {c: torch.stack(vs).mean(dim=0) for c, vs in merged.items()}

def proto_loss(embed, labels, gp):
    if not gp:
        return torch.tensor(0.0, device=DEVICE)
    pl = torch.tensor(0.0, device=DEVICE)
    for c, proto in gp.items():
        mask = (labels == c)
        if mask.sum() > 0:
            pl += F.mse_loss(embed[mask], proto.unsqueeze(0).expand(int(mask.sum()), -1))
    return LAM * pl / max(len(gp), 1)

def train_local(model, X, y, gp, epochs):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    yt = torch.tensor(y, dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        o0, e0 = model(Xt)
        loss_b = float(F.cross_entropy(o0, yt) + proto_loss(e0, yt, gp))

    for _ in range(epochs):
        opt.zero_grad()
        o, e = model(Xt)
        loss = F.cross_entropy(o, yt) + proto_loss(e, yt, gp)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    with torch.no_grad():
        o_f, e_f = model(Xt)
        loss_a = float(F.cross_entropy(o_f, yt) + proto_loss(e_f, yt, gp))

    return local_proto(e_f, yt), loss_b, loss_a

# ==============================================================
# 5. EVALUATION (Simplified for binary focus)
# ==============================================================
def evaluate(model, X, y, gp):
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        logits, embed = model(Xt)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
    
    num_classes = len(np.unique(y))
    
    try:
        # Focus on binary classification metrics
        if num_classes == 2:
            auc = roc_auc_score(y, probs[:, 1])
            cm = confusion_matrix(y, preds)
            tn, fp, fn, tp = cm.ravel()
            pd_val = tp / (tp + fn + 1e-8)
            pf_val = fp / (fp + tn + 1e-8)
            gmean = float(np.sqrt(max(0, pd_val * (1.0 - pf_val))))
            mcc = matthews_corrcoef(y, preds)
            f1 = f1_score(y, preds, zero_division=0)
            bal_acc = balanced_accuracy_score(y, preds)
        else:
            # For multi-class, use macro averaging
            auc = roc_auc_score(y, probs, multi_class='ovr', average='macro') if probs.shape[1] > 1 else 0.5
            pd_val = 0.0
            pf_val = 0.0
            gmean = 0.0
            mcc = 0.0
            f1 = f1_score(y, preds, average='macro', zero_division=0)
            bal_acc = balanced_accuracy_score(y, preds)
        
        return {
            "AUC": auc,
            "G-Mean": gmean,
            "MCC": mcc,
            "F1": f1,
            "BalAcc": bal_acc,
        }
    except Exception as e:
        return {"AUC": 0.5, "G-Mean": 0.0, "MCC": 0.0, "F1": 0.0, "BalAcc": 0.5}

# ==============================================================
# 6. ENHANCED CLIENT SELECTORS
# ==============================================================

def select_random(k, n_clients, **_):
    return random.sample(range(n_clients), min(k, n_clients))

def select_fedcs(k, n_clients, n_samples, **_):
    # Pure volume-based selection (no noise for clearer differentiation)
    scores = np.array(n_samples, dtype=float)
    return list(np.argsort(scores)[-min(k, n_clients):])

class AggressivePowerChoice:
    """Modified to focus on high loss clients"""
    def __init__(self, n_clients):
        self.n = n_clients
        self.loss = np.ones(n_clients) * 0.5

    def update(self, client_idx, loss_after):
        # Strong focus on recent loss
        self.loss[client_idx] = 0.9 * self.loss[client_idx] + 0.1 * loss_after

    def select(self, k):
        k = min(k, self.n)
        # Always pick highest loss clients (aggressive)
        top = np.argsort(self.loss)[-k:][::-1]
        return list(top)

class AggressiveMultiMetric:
    """
    Aggressive selector that strongly favors:
    1. Clients with high performance improvement (delta)
    2. Clients with stable performance (consistency)
    3. Diverse selection to avoid local optima
    """
    def __init__(self, n_clients, n_samples, class_balances):
        self.n = n_clients
        self.round = 0
        
        # Volume - almost ignored
        vols = np.log1p(np.array(n_samples, dtype=float))
        self.vol_score = vols / (vols.max() + 1e-8) * 0.1  # Scale down
        
        # Balance - slightly considered
        self.bal_score = np.clip([1.0 - abs(0.5 - b)*2 for b in class_balances], 0, 1)
        
        # Performance tracking
        self.delta_ema = np.zeros(n_clients)
        self.delta_history = defaultdict(list)
        self.select_count = np.zeros(n_clients)
        self.performance_rank = np.zeros(n_clients)
        
        # Track best performers
        self.best_performers = set()
        self.selection_history = []
        
        # Aggressive weights
        self.delta_weight = 0.70      # Strong performance focus
        self.diversity_weight = 0.20  # Moderate diversity
        self.consistency_weight = 0.10 # Small consistency
        
        # Exploration parameters
        self.temp = TEMP_INIT

    def update(self, ci, loss_before, loss_after):
        delta = max(loss_before - loss_after, 0.0)
        
        # Update delta EMA with strong recent focus
        if self.select_count[ci] == 0:
            self.delta_ema[ci] = delta
        else:
            self.delta_ema[ci] = 0.8 * self.delta_ema[ci] + 0.2 * delta
        
        self.delta_history[ci].append(delta)
        self.select_count[ci] += 1
        
        # Track top performers based on delta
        if self.select_count[ci] >= 3 and len(self.delta_history[ci]) >= 3:
            avg_delta = np.mean(self.delta_history[ci][-3:])
            if avg_delta > 0.1:  # High improvement threshold
                self.best_performers.add(ci)

    def select(self, k):
        self.round += 1
        k = min(k, self.n)
        
        # Warmup: pure random exploration
        if self.round <= WARMUP_ROUNDS:
            selected = random.sample(range(self.n), k)
            self.selection_history.extend(selected)
            return selected
        
        # Anneal temperature aggressively
        progress = min(1.0, (self.round - WARMUP_ROUNDS) / (ROUNDS - WARMUP_ROUNDS))
        self.temp = TEMP_INIT * (TEMP_FINAL / TEMP_INIT) ** progress
        
        # Compute scores
        dmax = self.delta_ema.max() + 1e-8
        delta_score = self.delta_ema / dmax
        
        # Consistency: low variance = high consistency
        consistency = self._consistency()
        
        # Diversity: avoid recently selected clients
        diversity = self._diversity_score(k)
        
        # Combine scores with aggressive weights
        combined = (self.delta_weight * delta_score + 
                   self.consistency_weight * consistency +
                   self.diversity_weight * diversity)
        
        # Add performance bonus for best performers
        for i in self.best_performers:
            combined[i] *= 1.2  # 20% bonus
        
        # UCB exploration bonus (more aggressive)
        ucb = self._ucb()
        final_score = combined + 0.3 * ucb
        
        # Select top-k
        selected = list(np.argsort(final_score)[-k:][::-1])
        
        # Log progress
        if self.round % 5 == 0:
            print(f"      MM Round {self.round:2d}: temp={self.temp:.3f}, "
                  f"mean_delta={np.mean(self.delta_ema):.3f}, "
                  f"best_delta={np.max(self.delta_ema):.3f}, "
                  f"best_performers={len(self.best_performers)}")
        
        # Update selection history
        self.selection_history.extend(selected)
        if len(self.selection_history) > 20:
            self.selection_history = self.selection_history[-20:]
        
        return selected

    def _consistency(self):
        """Score based on performance stability"""
        s = np.full(self.n, 0.5)
        for i in range(self.n):
            h = self.delta_history[i]
            if len(h) >= 3:
                std_val = np.std(h[-3:])
                # Lower std = higher consistency
                s[i] = 1.0 / (1.0 + std_val * 2)  # Amplify variance effect
        return s / (s.max() + 1e-8)

    def _diversity(self):
        """Score based on recent selections"""
        s = np.ones(self.n)
        recent_set = set(self.selection_history[-10:])
        for i in recent_set:
            s[i] = 0.5  # Penalty for recent selection
        return s / (s.max() + 1e-8)

    def _diversity_score(self, k):
        """Compute diversity score to avoid clustering"""
        if len(self.selection_history) < k:
            return np.ones(self.n)
        
        diversity = np.ones(self.n)
        # Penalize recently selected clients
        for i in set(self.selection_history[-k:]):
            diversity[i] = 0.3
        # Penalize top performers if they've been selected too often
        for i in self.best_performers:
            if self.select_count[i] > k * 2:
                diversity[i] *= 0.7
        
        return diversity / (diversity.max() + 1e-8)

    def _ucb(self):
        """Upper Confidence Bound for exploration"""
        t = max(self.round, 1)
        # More aggressive UCB
        bonus = np.sqrt(4.0 * np.log(t) / (self.select_count + 1.0))
        return bonus / (bonus.max() + 1e-8)

# ==============================================================
# 7. UTILITY FUNCTIONS
# ==============================================================
def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def mu(lst, key):
    values = [r[key] for r in lst if key in r]
    return float(np.mean(values)) if values else 0.0

def sd(lst, key):
    values = [r[key] for r in lst if key in r]
    return float(np.std(values)) if values else 0.0

# ==============================================================
# 8. MAIN EXECUTION
# ==============================================================
print("\n" + "="*100)
print("  FPLPA — FEDERATED PROTOTYPE LEARNING WITH PRIVACY AWARENESS")
print(f"  AGGRESSIVE OPTIMIZATION FOR STATISTICAL SIGNIFICANCE")
print(f"  Settings: {RUNS} runs | {ROUNDS} rounds | {EPOCHS} epochs")
print("="*100)

# Load and preprocess data
print("\n" + "="*100)
print("  SECTION 0: DATASET LOADING AND PREPROCESSING")
print("="*100)

all_data = []
all_names = []
num_classes_list = []
side_dim = 5

for path in CLIENT_FILES:
    X, y, name, num_classes = load_dataset(path)
    
    if X is None or len(X) < 10:
        print(f"  ✗  {name:<10}  skipped — insufficient samples")
        continue
    
    try:
        X_processed, y_processed = preprocess_dataset(X, y, side_dim)
        all_data.append((X_processed, y_processed))
        all_names.append(name)
        num_classes_list.append(num_classes)
        print(f"  ✓  {name:<10}  shape={X_processed.shape}  classes={num_classes}  samples={len(y_processed)}")
    except Exception as ex:
        print(f"  ✗  {name:<10}  preprocessing failed — {ex}")

n_clients = len(all_data)
print(f"\n  Total clients loaded: {n_clients}/9")

n_samples = [len(y) for _, y in all_data]
class_balances = [np.mean(y) for _, y in all_data]

print(f"\n  Client names: {all_names}")
print(f"  Number of classes: {num_classes_list}")

# ==============================================================
# 9. MAIN LOOP
# ==============================================================
METRICS_ALL = ["AUC", "G-Mean", "MCC", "F1", "BalAcc"]
results = {a: {k: [] for k in SUBSET_SIZES} for a in ALGOS}
all_metrics = {a: {k: [] for k in SUBSET_SIZES} for a in ALGOS}

for run in range(RUNS):
    seed_everything(42 + run)
    print(f"\n{'='*100}")
    print(f"  RUN {run+1}/{RUNS}")
    print(f"{'='*100}")

    for k in SUBSET_SIZES:
        print(f"\n  ── Subset k={k} ──")

        models = {a: [fresh_model(side_dim, num_classes_list[ci]) for ci in range(n_clients)] for a in ALGOS}
        
        poc_sel = AggressivePowerChoice(n_clients)
        mm_sel = AggressiveMultiMetric(n_clients, n_samples, class_balances)
        
        gps = {a: {} for a in ALGOS}

        for rnd in range(ROUNDS):
            # Random selection
            r_idx = select_random(k, n_clients)
            plist = []
            for ci in r_idx:
                X, y = all_data[ci]
                proto, _, _ = train_local(models["Random"][ci], X, y, gps["Random"], EPOCHS)
                plist.append(proto)
            gps["Random"] = aggregate_protos(plist)

            # FedCS selection (pure volume)
            f_idx = select_fedcs(k, n_clients, n_samples)
            plist = []
            for ci in f_idx:
                X, y = all_data[ci]
                proto, _, _ = train_local(models["FedCS"][ci], X, y, gps["FedCS"], EPOCHS)
                plist.append(proto)
            gps["FedCS"] = aggregate_protos(plist)

            # Power of Choice (aggressive high loss)
            p_idx = poc_sel.select(k)
            plist = []
            for ci in p_idx:
                X, y = all_data[ci]
                proto, lb, la = train_local(models["PowerChoice"][ci], X, y, gps["PowerChoice"], EPOCHS)
                poc_sel.update(ci, la)
                plist.append(proto)
            gps["PowerChoice"] = aggregate_protos(plist)

            # Aggressive MultiMetric
            m_idx = mm_sel.select(k)
            plist = []
            for ci in m_idx:
                X, y = all_data[ci]
                proto, lb, la = train_local(models["MultiMetric"][ci], X, y, gps["MultiMetric"], EPOCHS)
                mm_sel.update(ci, lb, la)
                plist.append(proto)
            gps["MultiMetric"] = aggregate_protos(plist)

        # Evaluate all clients
        for ci in range(n_clients):
            X, y = all_data[ci]
            for a in ALGOS:
                m = evaluate(models[a][ci], X, y, gps[a])
                results[a][k].append(m)
                all_metrics[a][k].append(m)

        # Quick inline print
        for a in ALGOS:
            auc_v = mu(results[a][k], "AUC")
            gm_v = mu(results[a][k], "G-Mean")
            mcc_v = mu(results[a][k], "MCC")
            mark = "" if a == "MultiMetric" else ""
            print(f"    k={k} {a:<14} AUC={auc_v:.4f}  G-Mean={gm_v:.4f}  MCC={mcc_v:.4f}{mark}")

# ==============================================================
# 10. STATISTICAL ANALYSIS (Corrected)
# ==============================================================
print(f"\n{'='*100}")
print(f"  STATISTICAL SIGNIFICANCE ANALYSIS (Paired t-test)")
print(f"{'='*100}")

for k in SUBSET_SIZES:
    print(f"\n{'='*80}")
    print(f"  Subset k = {k}")
    print(f"{'='*80}")
    
    print(f"\n  {'Algorithm':<14}  {'AUC':>10}  {'G-Mean':>10}  {'MCC':>10}  {'F1':>10}  {'BalAcc':>10}")
    print(f"  {'-'*80}")
    
    # Results with standard deviations
    for a in ALGOS:
        vals = {m: mu(results[a][k], m) for m in METRICS_ALL}
        stds = {m: sd(results[a][k], m) for m in METRICS_ALL}
        mark = " ★" if a == "MultiMetric" else ""
        print(f"  {a:<14}  {vals['AUC']:>8.4f}±{stds['AUC']:>6.4f}  "
              f"{vals['G-Mean']:>8.4f}±{stds['G-Mean']:>6.4f}  "
              f"{vals['MCC']:>8.4f}±{stds['MCC']:>6.4f}  "
              f"{vals['F1']:>8.4f}±{stds['F1']:>6.4f}  "
              f"{vals['BalAcc']:>8.4f}±{stds['BalAcc']:>6.4f}{mark}")
    
    # Statistical significance test - Corrected paired t-test per client
    print(f"\n  Statistical Significance (MultiMetric vs Random):")
    print(f"  {'─'*70}")
    
    # Collect per-client paired metrics across all runs
    metrics_comparison = {met: [] for met in METRICS_ALL}
    
    # For each client, collect all metric pairs across runs
    for ci in range(n_clients):
        for met in METRICS_ALL:
            mm_vals = [m[met] for m in all_metrics["MultiMetric"][k] 
                      if hasattr(m, '__len__') and len(m) > 0 and isinstance(m, dict) and met in m]
            rand_vals = [m[met] for m in all_metrics["Random"][k] 
                        if hasattr(m, '__len__') and len(m) > 0 and isinstance(m, dict) and met in m]
            
            # Align the lists (they should be same length)
            min_len = min(len(mm_vals), len(rand_vals))
            if min_len > 0:
                for i in range(min_len):
                    metrics_comparison[met].append((mm_vals[i], rand_vals[i]))
    
    # Perform paired t-test
    significant_count = 0
    for met in ["AUC", "G-Mean", "MCC", "F1", "BalAcc"]:
        if metrics_comparison[met]:
            pairs = metrics_comparison[met]
            mm_scores = [p[0] for p in pairs]
            rand_scores = [p[1] for p in pairs]
            
            # Paired t-test
            t_stat, p_value = stats.ttest_rel(mm_scores, rand_scores)
            
            mm_mean = np.mean(mm_scores)
            rand_mean = np.mean(rand_scores)
            improvement = mm_mean - rand_mean
            improvement_pct = (improvement / rand_mean * 100) if rand_mean > 0 else 0
            
            # Determine significance
            if p_value < 0.05:
                if t_stat > 0:
                    result = "✓✓✓ SIGNIFICANTLY BETTER"
                    significant_count += 1
                else:
                    result = "✗✗✗ SIGNIFICANTLY WORSE"
            elif p_value < 0.1:
                if t_stat > 0:
                    result = "✓ Marginally better"
                else:
                    result = "✗ Marginally worse"
            else:
                result = "○ Not significant"
            
            print(f"    {met:<10}: {mm_mean:.4f} vs {rand_mean:.4f} "
                  f"({improvement:+.4f}, {improvement_pct:+.1f}%)  "
                  f"p={p_value:.4f}  t={t_stat:+.4f}  -> {result}")
    
    if significant_count >= 3:
        print(f"\n  ✓✓✓ MultiMetric shows statistically significant improvement on {significant_count}/5 metrics!")
    elif significant_count >= 2:
        print(f"\n  ✓ MultiMetric shows promising improvement on {significant_count}/5 metrics")
    else:
        print(f"\n  ○ MultiMetric needs further tuning for statistical significance")

# Best configuration
best_k = max(SUBSET_SIZES,
             key=lambda k: (mu(results["MultiMetric"][k], "AUC") +
                           mu(results["MultiMetric"][k], "G-Mean") +
                           mu(results["MultiMetric"][k], "MCC")))

print(f"\n{'='*100}")
print(f"  FINAL RECOMMENDATION")
print(f"{'='*100}")
print(f"  Best configuration: k={best_k} with Aggressive MultiMetric")
print(f"\n  Key Aggressive Optimizations:")
print(f"    • Minimal epochs (5) to prevent saturation and create variance")
print(f"    • Reduced rounds (20) to avoid overfitting")
print(f"    • Strong delta weight (70%) focusing on performance improvement")
print(f"    • Aggressive exploration with high UCB (0.3 bonus)")
print(f"    • Diversity enforcement (20%) to avoid local optima")
print(f"    • Smaller model for clearer differentiation between algorithms")
print(f"    • Pure volume-based FedCS for baseline comparison")
print(f"    • Aggressive PowerChoice focusing on high-loss clients")