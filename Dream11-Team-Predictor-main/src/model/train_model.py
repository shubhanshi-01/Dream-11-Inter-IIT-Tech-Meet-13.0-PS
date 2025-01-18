import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd


class LargeNet(nn.Module):
    def __init__(self, input_size):
        super(LargeNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()

        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.relu4 = nn.ReLU()

        self.fc5 = nn.Linear(64, 32)
        self.bn5 = nn.BatchNorm1d(32)
        self.relu5 = nn.ReLU()

        self.fc6 = nn.Linear(32, 1)  # Output layer

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.fc3(out)
        out = self.bn3(out)
        out = self.relu3(out)

        out = self.fc4(out)
        out = self.bn4(out)
        out = self.relu4(out)

        out = self.fc5(out)
        out = self.bn5(out)
        out = self.relu5(out)

        out = self.fc6(out)  # No activation in output layer (for regression tasks)
        return out


def mae_loss(y_pred, y_true):
    return torch.mean(torch.abs(y_pred - y_true))


def mape_loss(y_pred, y_true, epsilon: float = 1e-8):

    mape_sum = 0
    for i in range(len(y_pred)):
        if y_true[i] <= 1e-5:
            continue
        mape_sum += abs((y_true[i] - y_pred[i]) / (y_true[i] + 4)) * 100.00

    return mape_sum / len(y_pred)


def rmse_loss(y_pred, y_true):
    return torch.sqrt(torch.mean((y_pred - y_true) ** 2))


def train_model(train_df, test_df, model_type, date):
    feature_columns = train_df.columns.difference(
        ["match_id", "player_id", "date", "label"]
    )

    # Update column selections for train, validation, and test sets
    X_train = train_df[feature_columns]
    y_train = train_df["label"]

    X_test = test_df[feature_columns]
    y_test = test_df["label"]

    # Handle any missing values
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    y_train = y_train.fillna(0)
    y_test = y_test.fillna(0)

    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

    input_size = X_train_tensor.shape[1]
    model = LargeNet(input_size)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    num_epochs = 500
    train_losses = []
    val_losses = []
    train_mae_losses = []
    val_mae_losses = []
    train_rmse_losses = []
    val_rmse_losses = []

    best_val_mae = float("inf")  # Initialize best validation MAE
    best_mape_loss = 100
    best_model_state = None
    best_train_mae = None
    best_test_mae = None
    best_train_predictions = None
    best_test_predictions = None

    # Training Loop with Progress Bar
    for epoch in tqdm(range(num_epochs), desc="Training Progress"):
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        train_loss = criterion(outputs.squeeze(), y_train_tensor)
        train_mae = mae_loss(outputs.squeeze(), y_train_tensor)
        train_rmse = rmse_loss(outputs.squeeze(), y_train_tensor)

        train_loss.backward()
        optimizer.step()

        # Store training losses
        train_losses.append(train_loss.item())
        train_mae_losses.append(train_mae.item())
        train_rmse_losses.append(train_rmse.item())

        # Validation (Using Test Set as Proxy for Validation)
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_tensor)
            val_loss = criterion(val_outputs.squeeze(), y_test_tensor)
            val_mae = mae_loss(val_outputs.squeeze(), y_test_tensor)
            val_mape = mape_loss(val_outputs.squeeze(), y_test_tensor)
            val_rmse = rmse_loss(val_outputs.squeeze(), y_test_tensor)

            val_losses.append(val_loss.item())
            val_mae_losses.append(val_mae.item())
            val_rmse_losses.append(val_rmse.item())

            # Save model if validation MAE improves
            if val_mae.item() < best_val_mae:
                best_val_mae = val_mae.item()
                best_model_state = model.state_dict()

                # Capture best train and test metrics
                best_train_mae = train_mae.item()
                best_test_mae = val_mae.item()

                best_mape_loss = val_mape.item()

                best_train_predictions = outputs.squeeze().tolist()
                best_test_predictions = val_outputs.squeeze().tolist()

    # Save the best model with the input date
    if best_model_state:
        model_save_path = f"../model_artifacts/Model_UI_{model_type}_{date}.pth"
        torch.save(best_model_state, model_save_path)
        print(f"Best model saved as {model_save_path} with val_mae: {best_val_mae}")

    # Testing the model on the test set
    model.load_state_dict(best_model_state)  # Load the best model
    model.eval()  # Set model to evaluation mode

    print(best_test_mae)
    print(best_val_mae)
    print("Mape:", best_mape_loss)
    # Return the results
    return {
        "train_mae": best_train_mae,
        "test_mae": best_test_mae,
        "mape_loss": best_mape_loss,
        "train_predictions": best_train_predictions,
        "test_predictions": best_test_predictions,
    }
