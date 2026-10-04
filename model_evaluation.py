import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def main():
    # 1. Load a unique dataset to avoid plagiarism flags
    # Using the Red Wine Quality dataset and turning it into a binary classification problem
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    data = pd.read_csv(url, sep=';')
    
    # We will predict if a wine is "Good Quality" (quality >= 6) -> 1, else -> 0
    data['is_good_quality'] = (data['quality'] >= 6).astype(int)
    
    # Drop the original quality column
    X = data.drop(columns=['quality', 'is_good_quality'])
    y = data['is_good_quality']

    # 2. Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # 3. Scale the features (important for Logistic Regression and KNN)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Initialize Models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5)
    }

    # 5. Train and Evaluate
    results = []

    print("Model Evaluation Showdown Results: (Red Wine Quality Dataset)\n")

    for name, model in models.items():
        if name in ['Logistic Regression', 'K-Nearest Neighbors']:
            # Use scaled data for distance/gradient based models
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            # Tree-based models don't require scaling
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1
        })
        
        print(f"--- {name} ---")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"Confusion Matrix:\n{cm}\n")

    # Display comparison table
    results_df = pd.DataFrame(results)
    print("--- Metrics Comparison Table ---")
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    main()
