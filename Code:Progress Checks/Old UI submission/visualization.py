import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def visualize_all_in_one(csv_files, model_file="universal_model.pt"):
    device = torch.device("cpu")
    loaded_model = torch.jit.load(model_file, map_location=device)
    loaded_model.eval()

    plt.figure(figsize=(8,6))
    global_min_label = float('inf')
    global_max_label = float('-inf')

    for csv_idx, csv_file in enumerate(csv_files):
        df = pd.read_csv(csv_file)
        # Normalize labels to [0,1] (assuming max label is 50)
        labels = df['Label'].values / 50.0
        features = df.drop(columns=['Label']).values
        
        features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
        with torch.no_grad():
            preds = loaded_model(features_tensor).view(-1).cpu().numpy()

        data_dict = {}
        for lbl, pred_val in zip(labels, preds):
            data_dict.setdefault(lbl, []).append(pred_val)

        sorted_labels = sorted(data_dict.keys())
        mean_preds = [np.mean(data_dict[lbl]) for lbl in sorted_labels]
        std_preds = [np.std(data_dict[lbl]) for lbl in sorted_labels]

        global_min_label = min(global_min_label, min(sorted_labels))
        global_max_label = max(global_max_label, max(sorted_labels))

        plt.errorbar(
            sorted_labels,
            mean_preds,
            yerr=std_preds,
            fmt='o-',
            capsize=4,
            label=f"{csv_file}"
        )
    
    plt.plot(
        [global_min_label, global_max_label],
        [global_min_label, global_max_label],
        'k--',
        label="Ideal"
    )

    plt.title("Predicted vs. Labeled Stiffness (All CSVs in One Plot)")
    plt.xlabel("Normalized Labeled Stiffness Level")
    plt.ylabel("Predicted Stiffness Level")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    csv_files = [
        "adjusted_preprocessed_bella_data.csv",
        "adjusted_preprocessed_charison_data.csv",
        "adjusted_preprocessed_chloe_data.csv",
        "adjusted_preprocessed_kevin_data.csv",
        "adjusted_preprocessed_yue_data.csv"
    ]
    visualize_all_in_one(csv_files, model_file="universal_model.pt")
