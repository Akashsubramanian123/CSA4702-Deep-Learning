import numpy as np
import pandas as pd
 
def generate_dataset(n_students=100, seed=42):
    rng = np.random.default_rng(seed)
    attendance = np.clip(rng.normal(80, 10, n_students), 50, 100)
    assignment = np.clip(rng.normal(72, 12, n_students), 35, 100)
    internal = np.clip(0.4 * attendance + 0.5 * assignment + rng.normal(0, 6, n_students), 30, 100)
    performance = (0.25 * attendance + 0.30 * assignment + 0.45 * internal
                   + rng.normal(0, 2.5, n_students))
    df = pd.DataFrame({
        "Student_ID": [f"S{i:03d}" for i in range(1, n_students + 1)],
        "Attendance": attendance.round(1),
        "Assignment_Score": assignment.round(1),
        "Internal_Marks": internal.round(1),
        "Performance": performance.round(1),
    })
    return df
