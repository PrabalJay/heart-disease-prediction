import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data():
    # Load data
    df = pd.read_csv('data/raw/dataset.csv')
    
    # Handle outliers (example for cholesterol)
    q1 = df['cholesterol'].quantile(0.25)
    q3 = df['cholesterol'].quantile(0.75)
    iqr = q3 - q1
    df = df[(df['cholesterol'] >= q1 - 1.5*iqr) & (df['cholesterol'] <= q3 + 1.5*iqr)]
    
    # Split data
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale numerical features
    scaler = StandardScaler()
    num_cols = ['age', 'resting bp s', 'cholesterol', 'max heart rate', 'oldpeak']
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    
    # Save processed data
    X_train.to_csv('data/processed/X_train.csv', index=False)
    X_test.to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    preprocess_data()