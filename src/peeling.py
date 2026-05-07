import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier
from sklearn.metrics import roc_auc_score

def peeling_procedure(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    tau: int = 50,
    low_auc_threshold: float = 0.70,
    random_state: int = 42
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Implements the 'Peeling Procedure' algorithm to generate a 'Hard' dataset.
    """
    status = "Fail"
    et = ExtraTreesClassifier(random_state=random_state)
    abc = AdaBoostClassifier(random_state=random_state)
    
    # Use RandomState to avoid picking the exact same indices relative to the changing size
    rng = np.random.RandomState(random_state)
    
    X_train_current = np.asarray(X_train).copy()
    y_train_current = np.asarray(y_train).copy()
    
    while len(X_train_current) > tau:
        prev_len = len(X_train_current)
        
        df = pd.DataFrame(X_train_current)
        df['target'] = y_train_current
        
        # Stratified Reduction: randomly remove exactly 10% of samples from each class (i.e. keep 90%)
        df = df.groupby('target', group_keys=False).sample(frac=0.90, random_state=rng)
        
        if len(df) == prev_len:
            # Prevents infinite loop if class size is too small to be reduced by frac=0.9
            break
            
        X_train_current = df.drop(columns=['target']).values
        y_train_current = df['target'].values
        
        if len(np.unique(y_train_current)) < 2:
            break
            
        # Model Training
        et.fit(X_train_current, y_train_current)
        abc.fit(X_train_current, y_train_current)
        
        # Evaluation
        if len(np.unique(y_test)) > 1:
            et_probs = et.predict_proba(X_test)
            abc_probs = abc.predict_proba(X_test)
            
            # Predict proba handling for binary vs multiclass
            if et_probs.shape[1] == 2:
                et_auc = roc_auc_score(y_test, et_probs[:, 1])
                abc_auc = roc_auc_score(y_test, abc_probs[:, 1])
            else:
                et_auc = roc_auc_score(y_test, et_probs, multi_class='ovr')
                abc_auc = roc_auc_score(y_test, abc_probs, multi_class='ovr')
        else:
            et_auc, abc_auc = 1.0, 1.0
            
        # Condition Check
        if et_auc <= low_auc_threshold and abc_auc <= low_auc_threshold:
            status = "Success"
            break
            
    return X_train_current, y_train_current, status
