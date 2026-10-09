import json
import numpy as np
from data_module import generate_dataset
from preprocess_module import preprocess, FEATURES
from model_module import LinearNeuron
from evaluate_module import evaluate
from plot_module import plot_preprocessing, plot_heatmap, plot_comparison
 
df = generate_dataset()
d = preprocess(df)
model = LinearNeuron(n_features=3)
history = model.train(d["X_tr_s"], d["y_tr_s"], lr=0.05, epochs=300)
 
pred_tr = d["y_scaler"].inverse_transform(model.forward(d["X_tr_s"]))
pred_te = d["y_scaler"].inverse_transform(model.forward(d["X_te_s"]))
pred_all = d["y_scaler"].inverse_transform(model.forward(d["X_all_s"]))
m_tr, m_te = evaluate(d["y_tr"], pred_tr), evaluate(d["y_te"], pred_te)
 
corr = df[FEATURES + ["Performance"]].corr().values
plot_preprocessing(d["X"], d["X_all_s"], "fig_preprocess.png")
plot_heatmap(corr, ["Attendance", "Assignment", "Internal", "Performance"], "fig_heatmap.png")
plot_comparison(d["y_te"].ravel(), pred_te.ravel(), "fig_compare.png")
 
desc = df[FEATURES + ["Performance"]].describe().T
out = {
 "sample": df.head(10).values.tolist(),
 "desc": [[i, *[round(float(v), 2) for v in r[["count","mean","std","min","25%","50%","75%","max"]]]] for i, r in desc.iterrows()],
 "result": [[df.Student_ID[i], float(df.Performance[i]), round(float(pred_all[i,0]),1), round(float(pred_all[i,0]-df.Performance[i]),1)] for i in range(10)],
 "m_tr": m_tr, "m_te": m_te, "W": d["y_scaler"].scale_.tolist(), "w": model.W.ravel().tolist(), "b": model.b.tolist(),
 "loss0": history[0], "lossN": history[-1]}
json.dump(out, open("results.json", "w"), indent=1)
print(json.dumps(out, indent=1)[:2500])
