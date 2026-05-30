import yaml
import numpy as np
import pandas as pd
import warnings
from pathlib import Path

# Suppress expected benchmark warnings (Constant features in ANOVA, etc.)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif, SelectFromModel

from src.data_preprocessing import build_dataset_paths, load_dataset_xy, extract_xy, encode_labels

def evaluate_etree_configs(X_tr, y_tr, X_te, y_te, seed=42):
    configs = [
        {"max_depth": 3, "n_estimators": 100},
        {"max_depth": 3, "n_estimators": 200},
        {"max_depth": 5, "n_estimators": 100},
    ]
    
    best_auc = -1.0
    best_model = None
    auc_list = []
    if len(np.unique(y_te)) <= 1:
        return 1.0, None, []
    
    for cfg in configs:
        et = ExtraTreesClassifier(random_state=seed, **cfg)
        et.fit(X_tr, y_tr)
        probs = et.predict_proba(X_te)
        
        if probs.shape[1] == 2:
            auc = roc_auc_score(y_te, probs[:, 1])
        else:
            auc = roc_auc_score(y_te, probs, multi_class='ovr')
        auc_list.append(auc)
        if auc > best_auc:
            best_auc = auc
            best_model = et
            
    return best_auc, best_model, auc_list

def get_best_fs_features(X_tr, y_tr, X_te, y_te, k=60, seed=42):
    k_actual = min(k, X_tr.shape[1])
    
    methods = [
        ("UnivariateFS", SelectKBest(score_func=f_classif, k=k_actual)),
        ("ExtraTrees", SelectFromModel(ExtraTreesClassifier(n_estimators=100, random_state=seed), max_features=k_actual)),
        ("LinearSVC", SelectFromModel(LinearSVC(penalty='l1', dual=False, random_state=seed, max_iter=50000), max_features=k_actual))
    ]
    
    best_auc = -1.0
    best_indices = None
    
    for name, selector in methods:
        selector.fit(X_tr, y_tr)
        indices = selector.get_support(indices=True)
        
        # Fallback if no features selected
        if len(indices) == 0:
            indices = np.arange(k_actual)
            
        X_tr_fs = X_tr[:, indices]
        X_te_fs = X_te[:, indices]
        
        auc, _, _ = evaluate_etree_configs(X_tr_fs, y_tr, X_te_fs, y_te, seed=seed)
        if auc > best_auc:
            best_auc = auc
            best_indices = indices
            
    return best_auc, best_indices

def run_peeling_experiment(X, y, dataset_name, seed=42, tau=50,     low_auc_threshold=0.70):
    print(f"\n--- Running Peeling Experiment on {dataset_name} ---")
    print(f"Initial dataset size: {X.shape[0]} samples, {X.shape[1]} features")
    y, _ = encode_labels(y)
    
    # Step 1: Initial Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.30, stratify=y, random_state=seed
    )
    
    auc_initial, _, _ = evaluate_etree_configs(X_train, y_train, X_test, y_test, seed=seed)
    print(f"Step 1: Initial AUC on X_test: {auc_initial:.4f}")
    if auc_initial <= low_auc_threshold:
        print(f"Warning: Initial AUC {auc_initial:.4f} is <= {low_auc_threshold}. Expected > {low_auc_threshold}.")
        
    # Step 2: Ground Truth Recovery
    auc_gt_initial, F_best = get_best_fs_features(X_train, y_train, X_test, y_test, k=60, seed=seed)
    print(f"Step 2: Ground Truth AUC (Initial Dataset): {auc_gt_initial:.4f} with {len(F_best)} features")
    
    # Step 3: The Peeling Loop
    print("Step 3: Executing Peeling Loop...")
    status = "Fail"
    X_train_peeled = np.asarray(X_train).copy()
    y_train_peeled = np.asarray(y_train).copy()
    rng = np.random.RandomState(seed)
    
    while len(X_train_peeled) > tau:
        prev_len = len(X_train_peeled)
        
        df = pd.DataFrame(X_train_peeled)
        df['target'] = y_train_peeled
        df = df.groupby('target', group_keys=False).sample(frac=0.90, random_state=rng)
        
        if len(df) == prev_len:
            break
            
        X_train_peeled = df.drop(columns=['target']).values
        y_train_peeled = df['target'].values
        
        if len(np.unique(y_train_peeled)) < 2:
            break
            
        et_auc, _, _ = evaluate_etree_configs(X_train_peeled, y_train_peeled, X_test, y_test, seed=seed)
        
        abc = AdaBoostClassifier(random_state=seed)
        abc.fit(X_train_peeled, y_train_peeled)
        if len(np.unique(y_test)) > 1:
            abc_probs = abc.predict_proba(X_test)
            if abc_probs.shape[1] == 2:
                abc_auc = roc_auc_score(y_test, abc_probs[:, 1])
            else:
                abc_auc = roc_auc_score(y_test, abc_probs, multi_class='ovr')
        else:
            abc_auc = 1.0
            
        if et_auc <= low_auc_threshold and abc_auc <= low_auc_threshold:
            status = "Success"
            break
            
    print(f"Peeling Status: {status} (Remaining samples: {len(X_train_peeled)})")
    
    # Step 4: Validation & Results Output
    if len(np.unique(y_train_peeled)) > 1:
        # 1. Peeled AUC
        auc_peeled, _, _ = evaluate_etree_configs(X_train_peeled, y_train_peeled, X_test, y_test, seed=seed)
            
        # 2. Peeled + FS AUC
        auc_peeled_fs, _ = get_best_fs_features(X_train_peeled, y_train_peeled, X_test, y_test, k=60, seed=seed)
            
        # 3. Peeled + GT AUC
        X_train_peeled_gt = X_train_peeled[:, F_best]
        X_test_gt = X_test[:, F_best]
        auc_peeled_gt, _, _ = evaluate_etree_configs(X_train_peeled_gt, y_train_peeled, X_test_gt, y_test, seed=seed)
    else:
        auc_peeled = 0.5
        auc_peeled_fs = 0.5
        auc_peeled_gt = 0.5
        print("Warning: Only 1 class remaining in peeled set.")

    print("\n--- Final Results Benchmark Table ---")
    print(f"{'Dataset':<15} | {'Base Peeled AUC':<17} | {'Peeled+FS AUC':<15} | {'Peeled+GT AUC':<15}")
    print("-" * 70)
    print(f"{dataset_name:<15} | {auc_peeled:<17.4f} | {auc_peeled_fs:<15.4f} | {auc_peeled_gt:<15.4f}")
    
    return {
        "Dataset": dataset_name,
        "AUC": auc_initial,
        "+FS": auc_gt_initial,
        "+Peeled": auc_peeled,
        "+Peeled FS": auc_peeled_fs,
        "+Peeled GT": auc_peeled_gt,
        "Size": len(X_train_peeled)
    }

def main():
    with open("config.yml", "r") as f:
        config_data = yaml.safe_load(f)
        
    seed = config_data.get("seed", 42)
    tau = config_data.get("peeling_tau", 50)
    low_auc = config_data.get("peeling_low_auc_threshold", 0.70)
    
    target_datasets = ["RELATHE", "SMK-CAN-187"]
    data_root = Path("data")
    dataset_paths = build_dataset_paths(data_root)
    
    results = []
    
    for ds_name in target_datasets:
        if ds_name in dataset_paths and dataset_paths[ds_name].exists():
            X, y = load_dataset_xy(dataset_paths[ds_name])
            X, y = extract_xy((X, y))
            metrics = run_peeling_experiment(X, y, ds_name, seed=seed, tau=tau, low_auc_threshold=low_auc)
            results.append(metrics)
        else:
            print(f"Dataset {ds_name} not found at {dataset_paths.get(ds_name)}")
            
    if results:
        df_results = pd.DataFrame(results)
        date_and_time = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        csv_path = output_dir / f"peeling_experiment_results_{date_and_time}.csv"
        df_results.to_csv(csv_path, index=False)
        print(f"\nAll results saved successfully to {csv_path}")

if __name__ == "__main__":
    main()
