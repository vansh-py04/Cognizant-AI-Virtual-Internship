import pandas as pd
import numpy as np 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

def load_file(path):
    df = pd.read_csv(path)
    df.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')
    return df

def target_and_predictors(data : pd.DataFrame=None,target="	estimated_stock_pct"):
    if target not in data.columns:
        raise Exception(f"Target: {target} is not present in the data")
    
    X = data.drop(columns=[target])
    y = data[target]
    return X, y

def train(X : pd.DataFrame=None, y: pd.Series=None):
    model = RandomForestRegressor(random_state=42)
    scaler = StandardScaler()

    x_train, x_test, y_train, y_test = train_test_split(X,y,random_state=42)
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    model.fit(x_train,y_train)
    pred = model.predict(x_test)
    error = mean_absolute_error(y_test,pred)
    print(f"MAE of Model:{error:.3f}")

def run():
    df = load_file()
    X,y = target_and_predictors(data=df)
    train(X,y)