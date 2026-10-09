import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
 
FEATURES = ["Attendance", "Assignment_Score", "Internal_Marks"]
 
def preprocess(df, test_size=0.2, seed=42):
    X = df[FEATURES].values            # feature matrix (100 x 3)
    y = df[["Performance"]].values     # target vector (100 x 1)
    idx = np.arange(len(df))
    X_tr, X_te, y_tr, y_te, i_tr, i_te = train_test_split(
        X, y, idx, test_size=test_size, random_state=seed)
    x_scaler, y_scaler = StandardScaler(), StandardScaler()
    X_tr_s = x_scaler.fit_transform(X_tr)
    X_te_s = x_scaler.transform(X_te)
    y_tr_s = y_scaler.fit_transform(y_tr)
    X_all_s = x_scaler.transform(X)
    return dict(X=X, y=y, X_tr=X_tr, X_te=X_te, y_tr=y_tr, y_te=y_te,
                X_tr_s=X_tr_s, X_te_s=X_te_s, y_tr_s=y_tr_s,
                X_all_s=X_all_s, i_tr=i_tr, i_te=i_te,
                x_scaler=x_scaler, y_scaler=y_scaler)
