import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import train_test_split


def split_datasets(X, y):
    n_samples = len(X)

    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    split_idx = int(0.67 * n_samples)
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_train = y[train_indices]
    y_train = y[train_indices]


def mono_log(zipcode):
    census_data = pd.read_csv("ressources/solar_system_census.csv")
    planets_data = pd.read_csv("ressources/solar_system_census_planets.csv")
    census_data.rename( columns={'Unnamed: 0':'index'}, inplace=True )
    planets_data.rename( columns={'Unnamed: 0':'index'}, inplace=True )
    
    merged_data = pd.merge(planets_data, census_data, on='index')
    
    X = merged_data[['weight', 'height', 'bone_density']]
    # Create the y array: 1 if planet is favorite, else 0
    y = np.where(merged_data['Origin'] == zipcode, 1, 0)

    # X_train, X_test, y_train, y_test = split_datasets(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
    

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