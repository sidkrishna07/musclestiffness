# debug_training.py

# ==========================================
# In[1]: Imports & Basic Setup
# ==========================================
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# For printing float arrays nicely
np.set_printoptions(precision=4, suppress=True)


# ==========================================
# In[2]: Verify CSV Data
# ==========================================
user_csvs = [
    "adjusted_preprocessed_bella1_data.csv",
    "adjusted_preprocessed_bella2_data.csv",
    "adjusted_preprocessed_charison1_data.csv",
    "adjusted_preprocessed_charison2_data.csv",
    "adjusted_preprocessed_chloe1_data.csv",
    "adjusted_preprocessed_chloe2_data.csv",
    "adjusted_preprocessed_kevin1_data.csv",
    "adjusted_preprocessed_kevin2_data.csv",
    "adjusted_preprocessed_yue1_data.csv",
]
universal_csv = "adjusted_preprocessed_all_data.csv"

def check_csv(csv_file: str) -> pd.DataFrame:
    """Load and verify each CSV's shape, columns, and label distribution."""
    print(f"\n=== Checking CSV: {csv_file} ===")
    if not os.path.exists(csv_file):
        print("  --> File does NOT exist!")
        return pd.DataFrame()  # return empty
    df = pd.read_csv(csv_file)
    
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    if "Label" not in df.columns:
        print("  --> No 'Label' column found!")
    else:
        print("  Label distribution:")
        print(df["Label"].describe())
        print(df["Label"].value_counts())
    print("  Head:")
    print(df.head(3))
    return df

def combine_csvs(csv_list, output_csv: str) -> pd.DataFrame:
    """Merges the given CSV files into one big DataFrame."""
    dfs = []
    for f in csv_list:
        if os.path.exists(f):
            df_temp = pd.read_csv(f)
            dfs.append(df_temp)
        else:
            print(f"WARNING: {f} not found. Skipping.")
    if len(dfs) == 0:
        print("No CSVs found, returning empty DataFrame.")
        return pd.DataFrame()
    
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.to_csv(output_csv, index=False)
    print(f"Merged {len(dfs)} CSVs into '{output_csv}' (original list had {len(csv_list)}).")
    return combined_df

# Check each CSV individually
for f in user_csvs:
    _ = check_csv(f)

# Merge them into one
if not os.path.exists(universal_csv):
    df_merged = combine_csvs(user_csvs, universal_csv)
else:
    print(f"\n'{universal_csv}' already exists, skipping merge...")
    df_merged = pd.read_csv(universal_csv)

print("\n=== Checking Final Merged CSV ===")
if not df_merged.empty:
    print("Shape:", df_merged.shape)
    print("Columns:", list(df_merged.columns))
    if "Label" in df_merged.columns:
        print("Label distribution in merged file:")
        print(df_merged["Label"].describe())
        print(df_merged["Label"].value_counts())
    print("First 5 rows of merged DF:")
    print(df_merged.head(5))
else:
    print("Merged DataFrame is empty. Check warnings above.")


# ==========================================
# In[3]: Model & Training with Debug
# ==========================================

# Hyperparameter Grid
HYPERPARAM_GRID = {
    'lr': [0.0005, 0.001],
    'dropout': [0.05, 0.1],
    'weightdecay': [1e-5, 1e-4],
}
EPOCHS = 80
STEP_SIZE = 60
GAMMA = 0.5
K_FOLDS = 5

# A simpler model: final layer without Sigmoid
# (If you want to keep the Sigmoid, re-add "nn.Sigmoid()" in the final layer.)
class MuscleStiffnessModel(nn.Module):
    def __init__(self, input_dim, dropout):
        super(MuscleStiffnessModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.network(x)

def train_and_eval(train_loader, test_loader, input_dim, lr, dropout, wd):
    model = MuscleStiffnessModel(input_dim, dropout)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    # We'll track final epoch's training loss for debug
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        # Optional: print a few epochs to see if loss is going down
        if (epoch + 1) % 20 == 0:
            avg_ep_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{EPOCHS} => Train Loss: {avg_ep_loss:.4f}")

    # Evaluate on test set
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        # We'll also print a sample prediction
        for i, (batch_features, batch_labels) in enumerate(test_loader):
            preds = model(batch_features)
            loss = criterion(preds, batch_labels)
            test_loss += loss.item()
            # Print debug on first batch only
            if i == 0:
                print("DEBUG: sample preds =>", preds[:5].squeeze().tolist())
                print("DEBUG: sample labels =>", batch_labels[:5].squeeze().tolist())
        avg_test_loss = test_loss / len(test_loader)
    return avg_test_loss, model

def crossval_search(data_file, hyperparams):
    print(f"\n=== Cross-validation Search on {data_file} ===")
    df = pd.read_csv(data_file)
    features = df.drop(columns=['Label']).values
    # Normalize labels to [0,1] (assuming max label is 50)
    labels = df['Label'].values / 50.0

    features_tensor = torch.tensor(features, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32).view(-1, 1)

    input_dim = features_tensor.shape[1]
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)

    best_loss = float('inf')
    best_combo = {}
    best_model = None

    for lr in hyperparams['lr']:
        for dropout in hyperparams['dropout']:
            for wd in hyperparams['weightdecay']:
                fold_losses = []
                last_fold_model = None
                for train_idx, test_idx in kf.split(features_tensor):
                    x_train, x_test = features_tensor[train_idx], features_tensor[test_idx]
                    y_train, y_test = labels_tensor[train_idx], labels_tensor[test_idx]

                    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=16, shuffle=True)
                    test_loader  = DataLoader(TensorDataset(x_test,  y_test),  batch_size=16, shuffle=False)

                    test_loss, trained_m = train_and_eval(train_loader, test_loader, input_dim, lr, dropout, wd)
                    fold_losses.append(test_loss)
                    last_fold_model = trained_m

                mean_loss = np.mean(fold_losses)
                print(f"  [lr={lr}, dropout={dropout}, wd={wd}] CV Loss = {mean_loss:.4f}")
                if mean_loss < best_loss:
                    best_loss = mean_loss
                    best_combo = {'lr': lr, 'dropout': dropout, 'weightdecay': wd}
                    best_model = last_fold_model

    print(f"\nBest combo found: {best_combo}, Loss={best_loss:.4f}")
    return best_combo, best_loss, best_model

def train_final_universal(data_file, best_combo):
    print(f"\n=== Final Training on entire dataset: {data_file} ===")
    df = pd.read_csv(data_file)
    features = df.drop(columns=['Label']).values
    labels = df['Label'].values / 50.0

    features_tensor = torch.tensor(features, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32).view(-1, 1)

    input_dim = features_tensor.shape[1]
    model = MuscleStiffnessModel(input_dim, best_combo['dropout'])
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=best_combo['lr'], weight_decay=best_combo['weightdecay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    dataset = TensorDataset(features_tensor, labels_tensor)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for batch_features, batch_labels in data_loader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()

        if (epoch+1) % 10 == 0:
            avg_loss = running_loss / len(data_loader)
            print(f"  Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

    # Save final universal model
    torchscript_model = torch.jit.script(model)
    torchscript_model.save("universal_model.pt")
    print("Saved final universal model as 'universal_model.pt'!")


# ==========================================
# In[4]: Run Crossval & Train
# ==========================================
if __name__ == "__main__":
    if df_merged.empty:
        print("No data to train on. Exiting.")
    else:
        best_params, best_loss, _ = crossval_search(universal_csv, HYPERPARAM_GRID)
        train_final_universal(universal_csv, best_params)
