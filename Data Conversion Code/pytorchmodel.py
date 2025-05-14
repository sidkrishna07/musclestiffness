import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
import os

# 1) RANDOM SEED FOR REPRODUCIBILITY
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# 2) List of Individual CSVs to Combine
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

# 3) Merge All CSV into One
universal_csv = "adjusted_preprocessed_all_data.csv"

def combine_csvs(csv_list, output_csv):
    dfs = []
    for f in csv_list:
        df = pd.read_csv(f)
        dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.to_csv(output_csv, index=False)
    print(f"Merged {len(csv_list)} CSVs into '{output_csv}'")

# 4) Hyperparameter Grid
HYPERPARAM_GRID = {
    'lr': [0.0005, 0.001],
    'dropout': [0.05, 0.1],
    'weightdecay': [1e-5, 1e-4],
}
EPOCHS = 80
STEP_SIZE = 60
GAMMA = 0.5
K_FOLDS = 5

# 5) Model Definition (with Sigmoid for output normalization)
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
            nn.Linear(32, 1),
            nn.Sigmoid()  # Ensures output is in [0,1]
        )
    def forward(self, x):
        return self.network(x)

# 6) Train & Evaluate for CV
def train_and_eval(train_loader, test_loader, input_dim, lr, dropout, wd):
    model = MuscleStiffnessModel(input_dim, dropout)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    for epoch in range(EPOCHS):
        model.train()
        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            preds = model(batch_features)
            loss = criterion(preds, batch_labels)
            test_loss += loss.item()
    avg_test_loss = test_loss / len(test_loader)
    return avg_test_loss, model

def crossval_search(data_file, hyperparams):
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

if __name__ == "__main__":
    if not os.path.exists(universal_csv):
        combine_csvs(user_csvs, universal_csv)
    else:
        print(f"'{universal_csv}' already exists, skipping merge...")

    print(f"\n=== Hyperparam Search on {universal_csv} ===")
    best_params, best_loss, _ = crossval_search(universal_csv, HYPERPARAM_GRID)

    print("\n=== Final Training on entire combined dataset ===")
    train_final_universal(universal_csv, best_params)
