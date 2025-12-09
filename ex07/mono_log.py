import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split 
from my_logistic_regression import MyLogisticRegression as MyLR

# def plot_scatter(X_train, X_test, y_true, y_pred):
#     features = ['weight', 'height', 'bone_density']
#     plt.figure(figsize=(15, 5))
#     subplot_idx = 0
    
#     for i in range(3):
#         for j in range(i + 1, 3):
#             plt.subplot(1, 3, subplot_idx +1)
#             plt.scatter(X_train[:, i], X_train[:, j], c=y_true.ravel(), cmap='coolwarm', alpha=0.5, label='True Label ', marker='o')
#             plt.scatter(X_test[:, i], X_test[:, j], c=y_pred.ravel(), cmap='coolwarm', alpha=0.5, label='Predicted Label', marker='x')
#             plt.xlabel(features[i])
#             plt.ylabel(features[j])
#             plt.legend()
#             plt.title(f'{features[i]} vs {features[j]}')
#             subplot_idx += 1

#     plt.tight_layout()
#     plt.show()

def plot_scatter(X_train, X_test, y_train, y_pred):
    features = ['weight', 'height', 'bone_density']
    plt.figure(figsize=(15, 5))
    subplot_idx = 0

    X_all = np.concatenate((X_train, X_test), axis=0)
    y_all = np.concatenate((y_train, y_pred), axis=0)
    n_train = X_train.shape[0]
    is_train = np.arange(X_all.shape[0]) < n_train

    for i in range(3):
        for j in range(i + 1, 3):
            plt.subplot(1, 3, subplot_idx +1)
            plt.scatter(X_all[is_train, i], X_all[is_train, j], c=y_train.ravel(), cmap='coolwarm', alpha=0.5, label='True Label ', marker='o')
            plt.scatter(X_all[~is_train, i], X_all[~is_train, j], c=y_pred.ravel(), cmap='coolwarm', alpha=0.5, label='Predicted Label', marker='x')
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
    split_idx = int(0.80 * n_samples)
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
    x_features = merged_data.columns.tolist()
    print("x_features", x_features)
    
    X = merged_data[['weight', 'height', 'bone_density', 'Origin']]
    
    # 3. Select your favorite Space Zipcode and generate a 
    # new numpy.array to label each citizen
    y = np.where(merged_data['Origin'] == zipcode, 1, 0)

    # 2. Split the dataset into a training and a test set
    X_train, X_test, y_train, y_test = split_datasets(X, y)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    # 4. Train a logistic model to predict 
    # if a citizen comes from your favorite planet or not,
    # using your brand new label.
    n = X_train.shape[1] + 1
    thetas = np.random.randn(n, 1)
    print(f"\nHyperparameters:")
    print("thetas (original):", thetas)
    mylr = MyLR(thetas, alpha=0.001, max_iter=1000)
    mylr.fit_(X_train.values, y_train)
    y_hat = mylr.predict_(X_test.values)

    print("thetas (optimized):", mylr.theta)
    print("loss:", mylr.vec_log_loss_(y_test, y_hat))

    # 5. Calculate and display the fraction of correct predictions 
    # over the total number of predictions based on the test set.
    binary_pred = (y_hat >= 0.5).astype(int)
    correct_pred = np.sum(binary_pred == y_test)
    total_pred = y_test.shape[0]
    fraction_correct = correct_pred / total_pred
    print(f"Fraction pred of correct pred: {correct_pred}/{total_pred}={fraction_correct}%")

    # 6. Plot 3 scatter plots (one for each pair of citizen features) with the dataset and the
    # final prediction of the model.
    plot_scatter(X_train.values, X_test.values, y_train, y_test)


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