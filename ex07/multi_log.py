import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
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
    
def split_datasets(X):
    print(X.head())
    n_samples = len(X)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    split_idx = int(0.80 * n_samples)
    print(split_idx)
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    X_train = X.iloc[train_indices, :-1]
    X_test = X.iloc[test_indices, :-1]
    y_train = X.iloc[train_indices, -1:]
    y_test = X.iloc[test_indices, -1:]

    return X_train, X_test, y_train, y_test

def label_y(y, zipcode):
    y_ = np.zeros((y.shape[0], 1))
    y_[np.where(y.values == int(zipcode))] = 1
    y_labelled = y_.reshape(-1, 1)
    return y_labelled

def print_classifiers(classifiers):
    for classif, zipcode in zip(classifiers, range(4)):
        print("Planet number:", zipcode)
        print("thetas:", classif.theta)
        print("alpha:", classif.alpha)
        print("max_iter:", classif.max_iter)


def multi_log():
    census_data = pd.read_csv("ressources/solar_system_census.csv")
    planets_data = pd.read_csv("ressources/solar_system_census_planets.csv")
    census_data.rename( columns={'Unnamed: 0':'index'}, inplace=True )
    planets_data.rename( columns={'Unnamed: 0':'index'}, inplace=True )
    
    merged_data = pd.merge(planets_data, census_data, on='index')
    x_features = merged_data.columns.tolist()
    print("x_features", x_features)
    
    X = merged_data[['weight', 'height', 'bone_density', 'Origin']]

    # 1. Split the dataset into a training and a test set.
    X_train, X_test, y_train, y_test = split_datasets(X)

    print("X_TRAIN:", X_train.head())
    print("X_test:", X_test.head())
    print("y_train:", y_train.head())
    print("y_test:", y_test.head())

    # 2. Train 4 logistic regression classifiers to discriminate 
    # each class from the others (the way you did in part one).
    classifiers = []
    predictions = np.zeros(y_test.shape)

    for zipcode in range(4):
        y_labelled_train = label_y(y_train, zipcode)
        y_labelled_test = label_y(y_test, zipcode)
        n = X_train.shape[1] + 1
        thetas = np.random.randn(n, 1)
        mylr = MyLR(thetas, alpha=0.001, max_iter=1000)
        mylr.fit_(X_train.values, y_labelled_train)
        classifiers.append(mylr)

        y_hat = mylr.predict_(X_test.values)
        binary_pred = (y_hat >= 0.5).astype(int)
        predictions[np.where(binary_pred == 1)] = zipcode
        correct_pred = np.sum(binary_pred == y_labelled_test)
        total_pred = y_labelled_test.shape[0]
        fraction_correct = correct_pred / total_pred
        print(f"Fraction pred of correct pred of {zipcode} planet: {correct_pred} / {total_pred} = {fraction_correct}%")
    predictions = predictions.reshape(-1, 1)

    print_classifiers(classifiers)
    
    # 3. Predict for each example the class according to 
    # each classifier and select the one
    # with the highest output probability score


    # 5. Calculate and display the fraction of correct predictions 
    # over the total number of predictions based on the test set.
    # binary_pref = (y_hat >= 0.5).astype(int)
    # correct_pred = np.sum(binary_pref == y_test)
    # total_pred = y_test.shape[0]
    # fraction_correct = correct_pred / total_pred
    # print(f"Fraction pred of correct pred: {correct_pred}/{total_pred}={fraction_correct}%")

    # # 6. Plot 3 scatter plots (one for each pair of citizen features) with the dataset and the
    # # final prediction of the model.
    # plot_scatter(X_train.values, X_test.values, y_train, y_test)


def main():
    try:
        assert len(sys.argv) == 1, "No argument needed."
        multi_log()
    except (ValueError, AssertionError) as error:
        print(error)


if __name__=="__main__":
    main()