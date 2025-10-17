# %% [markdown]
# Import the neccessary libraries

# %%
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt

# %% [markdown]
# Set plotting style for better visualization

# %%
plt.style.use("seaborn-v0_8-whitegrid")

# %% [markdown]
# Configuration and Constants

# %%
TICKER = "KO" # Coca-cola Company stock ticker
START_DATE = "2010-01-01" # Starting date for data acquisition
END_DATE = pd.to_datetime("today").strftime("%Y-%m-%d") # Today's date as end date
PREDICTION_DAYS = 1 # We aim to predict the stock price 1 day into the future

# %% [markdown]
# Data Loading and Acquistion

# %%
print(f"Fetching {TICKER} stock data from {START_DATE} to {END_DATE}......")

# Use yfinance to download historical stock data
try:
    df = yf.download(TICKER, start=START_DATE, end=END_DATE)
    print("Data loaded successfully!")
except Exception as e:
    print(f"Error fetching data: {e}")
    exit()

# %%
# Check if data was successfully loaded and is not empty
if df.empty:
    print(f"No data retrieved for {TICKER}. Exiting.")
    exit()

# %%
# Display the first few rows of the raw data
print("----- First 5 rows of the dataset -----")
df.head()

# %%
# Display the last few rows of the raw data
print("----- Last 5 rows of the dataset -----")
df.tail()

# %% [markdown]
# Data Preprocessing

# %%
# Check for missing values 
df_missing = df.isnull().sum()
print("Missing Values")
print(df_missing)

# %%
# Check for duplicated rows
df_duplicated = df.duplicated().sum()
print("Duplicated Rows")
print(df_duplicated)

# %% [markdown]
# Feature Engineering

# %%
# Create features based on the current day's trading data
features = df[["Open","High","Low","Close","Volume"]].copy()

# %%
# Target Variable Creation: We want to predict the "Close" price for the next day
# We shift the "Close" column upwards by -PREDICTION_DAYS (1 day) to align the future value
# (Target: y) with the current day's features (Input: X)
features["Target"] = features["Close"].shift(-PREDICTION_DAYS)

# %%
# Adding Simple moving averages (SMAs) as technical indicators
# SMAs smooth out price data to identify trend direction
features["SMA_5"] = features["Close"].rolling(window=5).mean()
features["SMA_20"] = features["Close"].rolling(window=20).mean()

# %%
# Drop rows with NaN values created by shifting and rolling window operations
# These NaNs are from the last row (Target) and first 20 rows (SMA_20)
features.dropna(inplace=True)

# %%
# Define the feature set (X) and the target variable (y)
X = features.drop("Target",axis=1) #  All columns except "Target" are features
y = features["Target"]

# %%
# Store the dates corresponding to the data points for visualization
dates = features.index.values

# %% [markdown]
# Data Splitting

# %%
# Split the data into training (80%) and testing (20%) sets.
# We do not shuffle time series data, as the order is crucial
# We use a fixed split point based on the index (time)
split_point = int(0.8 * len(X))
X_train,X_test =  X[:split_point], X[split_point:]
y_train,y_test = y[:split_point], y[split_point:]
dates_test = dates[split_point:] # Keep track of test dates for plotting

# %%
print(f"Training set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")

# %% [markdown]
# Data Scaling

# %%
# Standardize features by removing the mean and scaling to unit variance
# This is crucial for distance-based algorithms like SVR and Linear models with regulatization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test) # Use the fitted scaler from the training set

# %%
# Convert back to DataFrame for better visualization/debugging (optional, but good practice)
X_train_scaled = pd.DataFrame(X_train_scaled,columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled,columns=X_test.columns)

# %% [markdown]
# Model Definitions and Hyperparameter Grids

# %%
# Define the models and their respective hyperparameter search spaces for tuning

# 1. Linear Models (Ridge/Lasso/ElasticNet): Use "alpha" (regularization strength)
ridge_params = {'alpha': [0.1, 1.0, 10.0, 100.0]}
lasso_params = {'alpha': [0.001, 0.01, 0.1, 1.0]}

# %%
# 2. Decsion Tree: Use "max_depth" and "min_samples_split"
dt_params = {
    "max_depth" : np.arange(3,15), # Max depth of the  tree
    "min_smaples_split" : np.arange(2,10) # Minimum number of samples required to split an internal node
}

# %%
# 3. Random Forest: Use "n_estimators" (number of trees) and "max_depth"
rf_params = {
    "n_estimators" : np.arange(50,200,50), # Number of tree in the forest
    "max_depth" : np.arange(5,20),
    "min_samples_leaf" : [1,2,4]
}

# %%
# 4. SVR: Use "C" (regularization parameter) and "gamma" (kernel coefficient)
svr_params = {
    "C" : [0.1,1,10,100],
    "gamma" : ["scale","auto"], #  How much influence a single training example has
    "kernel" : ["rbf"]
}

# %% [markdown]
# Model Training and Comparison

# %%



