import pandas as pd
import numpy as np


def prepare_features(df, numerical_features, categorical_features):
   

    numerical_data = df[numerical_features].to_numpy()

    categorical_data = pd.get_dummies(
        df[categorical_features],
        drop_first=True
    ).to_numpy()

    features = np.hstack(
        (numerical_data, categorical_data)
    )

    return features