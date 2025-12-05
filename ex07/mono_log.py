import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from my_logistic_regression import MyLogisticRegression as MyLR

def plot_scatter(X, y_true, y_pred):
    features = ['weight', 'height', 'bone_density']
    plt.figure(figsize=(15, 5))
    subplot_idx = 0
    for i in range(3):
        for j in range(i + 1, 3):
            plt.subplot(1, 3, subplot_idx +1)
            plt.scatter(X[:, i], X[:, j], c=y_true.ravel(), cmap='coolwarm', alpha=0.5, label='True Label ', marker='o')
            plt.scatter(X[:, i], X[:, j], c=y_pred.ravel(), cmap='coolwarm', alpha=0.5, label='Predicted Label', marker='x')
            plt.xlabel(features[i])
            plt.ylabel(features[j])
            plt.legend()
            plt.title(f'{features[i]} vs {features[j]}')
            subplot_idx += 1

    plt.tight_layout()
    plt.show()

def split_datasets(X, y):
    n_samples = len(X)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    split_idx = int(0.50 * n_samples)
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test


def mono_log(zipcode):
    census_data = pd.read_csv("ressources/solar_system_census.csv")
    planets_data = pd.read_csv("ressources/solar_system_census_planets.csv")
    census_data.rename( columns={'Unnamed: 0':'index'}, inplace=True )
    planets_data.rename( columns={'Unnamed: 0':'index'}, inplace=True )
    
    merged_data = pd.merge(planets_data, census_data, on='index')
    x_features = census_data.columns.tolist()
    print("x_features", x_features)
    
    X = merged_data[['weight', 'height', 'bone_density']]
    # Create the y array: 1 if planet is favorite, else 0
    y = np.where(merged_data['Origin'] == zipcode, 1, 0)

    X_train, X_test, y_train, y_test = split_datasets(X, y)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    # print("X_train head():", X_train.head())
    # print("X_test head():", X_test.head())
    # print("y_train:", y_train)
    # print("y_test:", y_test)
    
    n = X_train.shape[1] + 1
    thetas = np.random.randn(n, 1)
    mylr = MyLR(thetas, alpha=0.001, max_iter=1000)
    mylr.fit_(X_train.values, y_train)
    y_hat = mylr.predict_(X_test.values)

    print(f"\nHyperparameters:")
    print("thetas (original):", thetas)
    print("thetas (optimized):", mylr.theta)
    print("loss:", mylr.vec_log_loss_(y_test, y_hat))

    # print("y_hat:", y_hat)
    mylr.fit_(X_train.values, y_train)
    plot_scatter(X_test.values, y_test, y_hat)
    # print("loss:", mylr.loss_(X_train.values, y_train))
    # print("fit", mylr.fit_(X_train.values, y_train))
    


def main():
    try:
        assert len(sys.argv) == 2, "You must enter 1 argument."
        if not sys.argv[1].startswith("--zipcode="):
            raise ValueError("Usage: python mono_log.py --zipcode=x (x: 0, 1, 2, or 3)")

        zipcode = int(sys.argv[1].split("=")[1])
        assert zipcode in {0, 1, 2, 3}, "The code must be 0, 1, 2 or 3."
        
        mono_log(zipcode)
    except (ValueError, AssertionError) as error:
        print(error)


if __name__=="__main__":
    main()