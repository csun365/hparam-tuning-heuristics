import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ann_result = pd.read_csv("results/ann_results.csv", index_col=0)
cnn_result = pd.read_csv("results/cnn_results.csv", index_col=0)
rnn_result = pd.read_csv("results/rnn_results.csv", index_col=0)
ann_tuning_result = pd.read_csv("results/ann_tuning_results_new.csv", index_col=0)
cnn_tuning_result = pd.read_csv("results/cnn_tuning_results_new.csv", index_col=0)
rnn_tuning_result = pd.read_csv("results/rnn_tuning_results_new.csv", index_col=0)

def plot_results(result, title_txt):
    plt.plot([np.max(result[result["cycle"]==i]["loss"]) for i in range(np.max(result["cycle"]) + 1)], linestyle="--", marker="o", label="max", c="royalblue")
    plt.plot([np.mean(result[result["cycle"]==i]["loss"]) for i in range(np.max(result["cycle"]) + 1)], linestyle="--", marker="o", label="mean", c="forestgreen")
    # plt.plot([np.median(result[result["cycle"]==i]["loss"]) for i in range(np.max(result["cycle"]) + 1)], linestyle="--", marker="o")
    plt.plot([np.min(result[result["cycle"]==i]["loss"]) for i in range(np.max(result["cycle"]) + 1)], linestyle="--", marker="o", label="min", c="orangered")
    plt.legend()
    plt.xlabel("Cycle Number")
    plt.ylabel("Loss")
    plt.title(title_txt)
    plt.show()

plot_results(ann_tuning_result, "ANN Tuning")
plot_results(cnn_tuning_result, "CNN Tuning")
plot_results(rnn_tuning_result, "RNN Tuning")
    
def plot_3D(result, title_txt):
  fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
  # fig = plt.figure(figsize=(6,6))
  
  # ax = fig.add_subplot(projection="3d")
  # s = np.argsort(np.argsort(result["loss"].values))
  
  if result.columns[0] == "learning_rate":
    # ax.scatter(np.log10(result[result.columns[0]]), result[result.columns[1]], result["loss"], c=result["loss"], cmap="YlOrRd", s=40)
    ax.stem(np.log10(result[result.columns[0]].values), result[result.columns[1]].values, result["loss"].values)
  else:
    # ax.scatter(result[result.columns[0]], result[result.columns[1]], result["loss"], c=result["loss"], cmap="YlOrRd", s=40)
    ax.stem(result[result.columns[0]].values, result[result.columns[1]].values, result["loss"].values)
  ax.set_xlabel(result.columns[0])
  ax.set_ylabel(result.columns[1])
  ax.set_zlabel("Loss")
  ax.set_title(title_txt)
  plt.show()

plot_3D(ann_result, "ANN Baseline Models")
plot_3D(cnn_result, "CNN Baseline Models")
plot_3D(rnn_result, "RNN Baseline Models")
  
p = sns.jointplot(data=ann_tuning_result, x=ann_tuning_result["num_units"], y=ann_tuning_result["dropout_rate"], hue="cycle", palette="viridis", joint_kws={"s": 50})
p.ax_joint.set_xlabel("Number of Hidden Units")
p.ax_joint.set_ylabel("Dropout Rate")
p.ax_joint.set_title("ANN Tuning", y=1.0, pad=-30)
plt.show()

p = sns.jointplot(data=cnn_tuning_result, x=np.log10(cnn_tuning_result["learning_rate"]), y=cnn_tuning_result["num_filters"], hue="cycle", palette="viridis", joint_kws={"s": 50})
p.ax_joint.set_xlabel("Learning Rate (log10)")
p.ax_joint.set_ylabel("Number of Filters")
p.ax_joint.set_title("CNN Tuning", y=1.0, pad=-30)
plt.show()

p = sns.jointplot(data=rnn_tuning_result, x=rnn_tuning_result["lookback"], y=rnn_tuning_result["num_units"], hue="cycle", palette="viridis", joint_kws={"s": 50})
p.ax_joint.set_xlabel("Lookback Length")
p.ax_joint.set_ylabel("Number of LSTM Units")
p.ax_joint.set_title("RNN Tuning", y=1.0, pad=-30)
plt.show()
  
sns.catplot(data=ann_tuning_result, x="cycle", y="loss", palette="mako", zorder=1)
sns.regplot(data=ann_tuning_result, x="cycle", y="loss", scatter=False, truncate=False, order=1, color=".2", ci=90)
plt.xlabel("Cycle Number")
plt.ylabel("Loss")
plt.title("ANN Tuning", pad=-20)
plt.tight_layout()
plt.show()

sns.catplot(data=ann_tuning_result, x="cycle", y="loss", palette="mako", zorder=1)
sns.regplot(data=ann_tuning_result, x="cycle", y="loss", scatter=False, truncate=False, order=1, color=".2", ci=90)
plt.xlabel("Cycle Number")
plt.ylabel("Loss")
plt.title("CNN Tuning", pad=-20)
plt.tight_layout()
plt.show()

sns.catplot(data=ann_tuning_result, x="cycle", y="loss", palette="mako", zorder=1)
sns.regplot(data=ann_tuning_result, x="cycle", y="loss", scatter=False, truncate=False, order=1, color=".2", ci=90)
plt.xlabel("Cycle Number")
plt.ylabel("Loss")
plt.title("RNN Tuning", pad=-20)
plt.tight_layout()
plt.show()
  
# pal = sns.cubehelix_palette(4, rot=-.25, light=.7)
# g = sns.FacetGrid(ann_tuning_result, row="cycle", hue="cycle", aspect=15, height=.5, palette=pal)
# g.map(sns.kdeplot, "loss",
#       bw_adjust=.5, clip_on=False,
#       fill=True, alpha=1, linewidth=1.5)
# g.map(sns.kdeplot, "loss", clip_on=False, color="w")
# # g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

# def label(x, color, label):
#   ax = plt.gca()
#   ax.text(0, .2, label, fontweight="bold", color=color,
#           ha="left", va="center", transform=ax.transAxes)

# g.map(label, "loss")
# # g.figure.subplots_adjust(hspace=0.25)
# g.set_titles("")
# g.set(yticks=[], ylabel="")
# g.despine(bottom=True, left=True)
# plt.show()
  
plt.scatter(ann_result["num_units"], ann_result["dropout_rate"], c=ann_result["loss"], cmap="summer", s = np.argsort(np.argsort(ann_result["loss"])[::-1]) ** 1.4)
cbar = plt.colorbar()
cbar.set_label("Loss")
plt.xlabel("Number of Hidden Units")
plt.ylabel("Dropout Rate")
plt.title("ANN Baseline Models")
plt.show()

plt.scatter(np.log10(cnn_result["learning_rate"][cnn_result["loss"] < 1]), cnn_result["num_filters"][cnn_result["loss"] < 1], c=cnn_result["loss"][cnn_result["loss"] < 1], cmap="summer", s = np.argsort(np.argsort(cnn_result["loss"][cnn_result["loss"] < 1])[::-1]) ** 1.4)
cbar = plt.colorbar()
cbar.set_label("Loss")
plt.xlabel("Learning Rate (Log)")
plt.ylabel("Number of Filters")
plt.title("CNN Baseline Models")
plt.show()

plt.scatter(rnn_result["lookback"], rnn_result["num_units"], c=rnn_result["loss"], cmap="summer", s = np.argsort(np.argsort(rnn_result["loss"])[::-1]) ** 1.4)
cbar = plt.colorbar()
cbar.set_label("Loss")
plt.xlabel("Lookback Length")
plt.ylabel("Number of LSTM Units")
plt.title("RNN Baseline Models")
plt.show()
  
p = sns.jointplot(x=ann_result["num_units"], y=ann_result["dropout_rate"], color="#4CB391")
p.ax_joint.set_xlabel("Number of Hidden Units")
p.ax_joint.set_ylabel("Dropout Rate")
p.ax_joint.set_title("ANN Baseline Distribution", pad=-50)
plt.tight_layout()
plt.show()

p = sns.jointplot(x=np.log10(cnn_result["learning_rate"]), y=cnn_result["num_filters"], color="#4CB391")
p.ax_joint.set_xlabel("Learning Rate (Log)")
p.ax_joint.set_ylabel("Number of Filters")
p.ax_joint.set_title("CNN Baseline Distribution", pad=-50)
plt.tight_layout()
plt.show()

p = sns.jointplot(x=rnn_result["lookback"], y=ann_result["num_units"], color="#4CB391")
p.ax_joint.set_xlabel("Lookback Length")
p.ax_joint.set_ylabel("Number of LSTM Units")
p.ax_joint.set_title("RNN Baseline Distribution", pad=-50)
plt.tight_layout()
plt.show()