import pandas as pd
import geopandas as gpd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from model.estimator import GARegressor
import time
import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import KFold
import pickle
from joblib import Parallel, delayed
from datetime import datetime
import os
import psutil

# Determine the number of cores and set up parallel processing
N_CORES = os.cpu_count() - 1

# Choose your parallelization strategy:
# Option 1: Maximize parallel trials, run CV serially within each trial
# N_CORES_CV = 1  # Serial CV within each trial
# N_CORES_TRIALS = 1  # Use all cores for parallel trials

# Option 2: Balance trials and CV (activate by commenting Option 1 and uncommenting below)
N_CORES_CV = 1  # Use 4 cores for CV folds
N_CORES_TRIALS = 2 # max(1, (N_CORES - 1) // (N_CORES_CV + 1))  # Divide cores evenly

print(f"Number of available cores: {N_CORES}")
print(f"Using {N_CORES_CV} cores for cross-validation")
print(f"Running {N_CORES_TRIALS} parallel trials")

# Add CPU monitoring function
def monitor_cpu_usage(interval=10):
    """Monitor CPU usage and log it periodically"""
    while True:
        cpu_percent = psutil.cpu_percent(interval=interval, percpu=True)
        avg_usage = sum(cpu_percent) / len(cpu_percent)
        with open("cpu_usage_log.txt", "a") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp} - Avg CPU: {avg_usage:.2f}% - Per Core: {cpu_percent}\n")

# Start monitoring in a background thread
import threading
monitor_thread = threading.Thread(target=monitor_cpu_usage, daemon=True)
monitor_thread.start()

# Load data
grid = pd.read_parquet('../grid_large_std_v2.parquet')

def calculate_midpoint(value):
    if value is None or value == "None":
        return np.nan
    try:
        if '-' in value:
            parts = value.split('-')
            # Convert each part to float after stripping whitespace
            low = float(parts[0].strip())
            high = float(parts[1].strip())
            return (low + high) / 2
        else:
            return float(value)
    except Exception:
        return np.nan

grid["Coarse_mid"] = grid.PercentCoarse.apply(calculate_midpoint)
numeric_cols = grid.select_dtypes(include=[np.number]).columns
filtered_numeric_cols = numeric_cols.difference(['latitude', 'longitude', 'Coarse_mid']).to_numpy()
tab_x = list(filtered_numeric_cols)
tab_l = ['latitude', 'longitude']
tab_y = ["Coarse_mid"]
df = grid[~grid.Coarse_mid.isna()]
X, y = df[tab_x + tab_l], df[tab_y]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("All data loaded.")

def objective(trial, n_split=5):
    """
    Optuna objective function that performs k-fold cross validation
    """
    trial_start = time.time()
    trial_id = trial.number
    
    # Log start of trial
    with open("trial_progress.txt", "a") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"Trial {trial_id} started at {timestamp}\n")
    
    # Suggest hyperparameters
    params = {
        'x_cols':        tab_x,
        'spa_cols':      tab_l,
        'y_cols':        tab_y,
        'attn_variant':  'MCPA',
        'd_model':       trial.suggest_categorical('d_model', [32, 64, 80]),
        'n_attn_layer':  trial.suggest_int('n_attn_layer', 1, 3),
        'idu_points':    trial.suggest_int('idu_points', 2, 8),
        'seq_len':       trial.suggest_categorical('seq_len', [64, 81, 100, 144, 256, 400]),
        'attn_dropout':  trial.suggest_float('attn_dropout', 0.01, 0.5),
        'attn_bias_factor': None,
        'reg_lin_dims':  trial.suggest_categorical('reg_lin_dims', [[1], [4, 1], [16, 1]]),
        'epochs':        trial.suggest_int('epochs', 3, 30),
        'lr':            5e-3,
        'batch_size':    8,
    }
    
    # K-Fold splitter
    kf = KFold(n_splits=n_split, shuffle=True, random_state=42)
    fold_splits = list(kf.split(X_train, y_train))
    
    def train_and_evaluate_fold(fold_idx):
        """Train a GARegressor on a fold and return its MAE"""
        trn_idx, val_idx = fold_splits[fold_idx]
        
        fold_start = time.time()
        with open("fold_progress.txt", "a") as f:
            f.write(f"Trial {trial_id}, Fold {fold_idx} starting\n")
        
        # Split data
        # trn_X, trn_y = X_train.iloc[trn_idx], y_train.iloc[trn_idx]
        # val_X, val_y = X_train.iloc[val_idx], y_train.iloc[val_idx]
        trn_X = X_train.iloc[trn_idx].copy()
        trn_y = y_train.iloc[trn_idx].copy()
        val_X = X_train.iloc[val_idx].copy()
        val_y = y_train.iloc[val_idx].copy()
        
        # Create and train the model
        model = GARegressor(**params)
        model.fit(
            X=trn_X[tab_x],
            l=trn_X[tab_l],
            y=trn_y
        )
        
        # Predict and calculate loss
        y_pred = model.predict(
            X=val_X[tab_x],
            l=val_X[tab_l]
        )
        fold_loss = mean_absolute_error(y_true=val_y, y_pred=y_pred)
        r2 = r2_score(val_y, y_pred)
        
        fold_time = time.time() - fold_start
        with open("fold_progress.txt", "a") as f:
            f.write(f"Trial {trial.params}, Fold {fold_idx} completed in {fold_time:.2f}s, MAE: {fold_loss:.6f}, R²: {r2:.6f}\n")
        
        return fold_loss
    
    # Run cross-validation folds - in parallel if N_CORES_CV > 1
    if N_CORES_CV > 1:
        # Use threading backend for nested parallelism to avoid conflicts
        fold_losses = Parallel(n_jobs=N_CORES_CV, backend="loky")(
            delayed(train_and_evaluate_fold)(fold_idx) 
            for fold_idx in range(n_split)
        )
    else:
        # Run serially
        fold_losses = [train_and_evaluate_fold(fold_idx) for fold_idx in range(n_split)]
    
    mean_loss = np.mean(fold_losses)
    trial_time = time.time() - trial_start
    
    # Log completion of trial
    with open("trial_progress.txt", "a") as f:
        f.write(f"Trial {trial_id} completed in {trial_time:.2f}s, Mean MAE: {mean_loss:.6f}\n")
        f.write(f"Parameters: {trial.params}\n")
        f.write("-" * 40 + "\n")
    
    return mean_loss

def logging_callback(study, trial):
    """Callback function for logging trial results"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Write trial details to the results file
    with open("results_v2.txt", "a") as f:
        f.write(f"Trial {trial.number} completed at {timestamp}\n")
        f.write(f"Trial loss (MAE): {trial.value}\n")
        f.write(f"Trial parameters: {trial.params}\n")
        f.write("-" * 40 + "\n")
    
    # Prepare trial data for the CSV file
    trial_data = {
        "Trial": trial.number,
        "Timestamp": timestamp,
        "MAE": trial.value
    }
    
    # Add all parameters to the dictionary
    trial_data.update(trial.params)
    
    csv_file = "trial_results_v2.csv"
    if os.path.exists(csv_file):
        temp = pd.read_csv(csv_file)
        # Append the new trial data (avoiding deprecated append method)
        temp = pd.concat([temp, pd.DataFrame([trial_data])], ignore_index=True)
    else:
        temp = pd.DataFrame([trial_data])
    
    # Save the updated DataFrame to CSV
    temp.to_csv(csv_file, index=False)
    
    # If this is the best trial so far, save it separately
    if study.best_trial.number == trial.number:
        with open("best_trial.txt", "w") as f:
            f.write(f"Best trial so far: #{trial.number}\n")
            f.write(f"Best MAE: {trial.value}\n")
            f.write(f"Best parameters: {trial.params}\n")

# Main optimization
if __name__ == "__main__":
    # Initialize files
    for filename in ["trial_progress.txt", "fold_progress.txt", "cpu_usage_log.txt", "results_v2.txt"]:
        with open(filename, "w") as f:
            f.write(f"=== Starting optimization at {datetime.now()} ===\n")
    
    # Create and configure the study
    sampler = TPESampler(seed=42)  # Set seed for reproducibility
    start_time = time.time()
    
    study = optuna.create_study(
        direction='minimize',
        study_name='ga-hp-optimized',
        sampler=sampler
    )
    
    # Run optimization with the chosen parallelization strategy
    print(f"Starting optimization with {N_CORES_TRIALS} parallel trials, each using {N_CORES_CV} cores for CV")
    study.optimize(
        objective, 
        n_trials=50, 
        n_jobs=N_CORES_TRIALS,
        callbacks=[logging_callback],
        gc_after_trial=True
    )
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Log final results
    best_params = study.best_params
    best_value = study.best_value
    best_trial = study.best_trial
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("results_v2.txt", "a") as f:
        f.write(f"\n\n=== FINAL RESULTS ===\n")
        f.write(f"Experiment completed at {timestamp}\n")
        f.write(f"Total elapsed time: {total_time:.2f}s ({total_time/3600:.2f} hours)\n")
        f.write(f"Best hyperparameters: {best_params}\n")
        f.write(f"Best overall loss (MAE): {best_value}\n")
        f.write(f"Best trial number: {best_trial.number}\n")
        f.write("=" * 60 + "\n")
    
    # Train a final model using the best hyperparameters
    try:
        print("Training final model with best hyperparameters...")
        
        # Create parameter dictionary with best hyperparameters
        final_params = {
            'x_cols': tab_x,
            'spa_cols': tab_l,
            'y_cols': tab_y,
            'attn_variant': 'MCPA',
            'attn_bias_factor': None,
            'lr': 5e-3,
            'batch_size': 8,
        }
        final_params.update(best_params)  # Add best hyperparameters
        
        # Create and train the final model
        final_model = GARegressor(**final_params)
        final_model.fit(X=X_train[tab_x], l=X_train[tab_l], y=y_train)
        
        # Evaluate on test set
        y_pred = final_model.predict(X=X_test[tab_x], l=X_test[tab_l])
        test_mae = mean_absolute_error(y_test, y_pred)
        test_r2 = r2_score(y_test, y_pred)
        
        # Save results and model
        with open("results_v2.txt", "w") as f:
            f.write(f"Test set MAE: {test_mae}\n")
            f.write(f"Test set R²: {test_r2}\n")
            f.write(f"Best hyperparameters: {best_params}\n")
        
        # Save the final model
        with open("final_model.pkl", "wb") as f:
            pickle.dump(final_model, f)
            
        print(f"Final model trained and saved. Test MAE: {test_mae}, Test R²: {test_r2}")
        
    except Exception as e:
        print(f"Error training final model: {e}")