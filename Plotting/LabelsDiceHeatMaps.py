import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import textwrap
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Your label name mapping
label_names = {
    1: "Abdominal solid organs",
    2: "GI tract",
    3: "Urinary_reproductive",
    4: "Respiratory system",            
    5: "Heart & vessels",
    6: "Head",
    7: "Spinal cord",
    8: "Spine",
    9: "Rib cages",
    10: "Appendicular skeleton",
    11: "Muscles",
}

overallcmap = "viridis"  


# Load your data (replace with your actual file path)
df = pd.read_csv("LabelDices.txt")

# Rename label columns
rename_dict = {f"label_{i}_dice": label_names[i] for i in label_names}
df = df.rename(columns=rename_dict)

# print the models found
print("Models found in data:")
print(df["model"].unique())

# make map for models names to more readable names
model_name_map = {
    "v1_dinov3_vits16": "UNetDino",
    "AG_fullV2_dinov3_vits16": "UNetDinoAttGate",
    "VanillaUNet_model": "VanillaUNet",
    "DinoEncdinov3_vits16": "DinoEnc",
}
df["model"] = df["model"].map(model_name_map)

# fix order that the models are plotted in
model_order = [ "DinoEnc", "UNetDino", "UNetDinoAttGate", "VanillaUNet"]
df["model"] = pd.Categorical(df["model"], categories=model_order, ordered=True)



label_cols = list(rename_dict.values())
train_sizes = sorted(df["TrainSize"].unique())
# flip train sizes for better visual order (largest at top)
train_sizes = train_sizes[::-1]
# put first last if it is 150.0
if 150.0 in train_sizes:
    train_sizes.remove(150.0)
    train_sizes.append(150.0)

def wrap_text(text, width=15):
    return "\n".join(textwrap.wrap(text, width))

def wrap_model(name):
    return "\n".join(textwrap.wrap(name, width=12))



wrapped_labels = [wrap_text(label, width=15) for label in label_cols]

# Create subplots (stacked vertically)
fig, axes = plt.subplots(
    nrows=len(train_sizes),
    ncols=1,
    figsize=(26, 5 * len(train_sizes)),   
    constrained_layout=True         
)

if len(train_sizes) == 1:
    axes = [axes]

vmin, vmax = 0, 1

for ax, ts in zip(axes, train_sizes):
    df_ts = df[df["TrainSize"] == ts]
    heatmap_data = df_ts.set_index("model")

    heatmap_data = heatmap_data.loc[model_order, label_cols]

    sns.heatmap(
        heatmap_data,
        ax=ax,
        cmap=overallcmap,
        vmin=vmin,
        vmax=vmax,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar=False
    )
    if ts == 150.0:
        ax.set_title(f"Train Size = 150 slices", fontsize=14, pad=10)
    else:
        ax.set_title(f"Train Size = {ts} %", fontsize=14, pad=10)


    labels = [wrap_model(label.get_text()) for label in ax.get_yticklabels()]
    ax.set_yticklabels(labels, fontweight="bold")

    ax.set_xticklabels(
        wrapped_labels,
        rotation=0,
        fontsize=11,
        fontweight="bold"
    )

# -----------------------------
# 5. Shared colorbar
# -----------------------------
sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=overallcmap)
sm.set_array([])

cbar = fig.colorbar(
    sm,
    ax=axes,
    orientation="vertical",
    pad=0.02
)
cbar.set_label("DICE Score", fontsize=12)

# -----------------------------
# 6. Apply wrapped x-labels once (bottom only)
# -----------------------------

#axes[-1].set_xticklabels(wrapped_labels, rotation=0, fontsize=11,fontweight="bold" )

# Improve spacing
#plt.subplots_adjust(bottom=0.18, hspace=0.35)
#plt.tight_layout()

# -----------------------------
# 7. Show or save
# -----------------------------
plt.savefig("LabelsDicesHeatmap.png", dpi=600, bbox_inches="tight")
plt.show()