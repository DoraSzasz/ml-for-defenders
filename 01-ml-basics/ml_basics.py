"""IP activity classifier - toy decision tree for the ML-for-hackers series.

Companion code for Post 1. Proof of concept, not a product.
Run from this directory after `pip install -r requirements.txt`.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

FEATURE_COLS = ["port_count", "avg_time_between_connections", "is_blacklisted"]


def main() -> None:
    df = pd.read_csv("ip_activity.csv")
    print(df.head())
    print("\nlabel counts:\n", df["label"].value_counts())

    le = LabelEncoder()
    df["label"] = le.fit_transform(df["label"])

    X = df[FEATURE_COLS]
    y = df["label"]

    X_train, _X_test, y_train, _y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    clf = DecisionTreeClassifier(max_depth=4, random_state=42)
    clf.fit(X_train, y_train)

    def predict_ip(port_count: int, avg_time: float, blacklisted: int) -> str:
        row = pd.DataFrame(
            [[port_count, avg_time, blacklisted]], columns=FEATURE_COLS
        )
        pred = clf.predict(row)
        return le.inverse_transform(pred)[0]

    print("\npredictions:")
    print("  55 ports, 1s gap, blacklisted ->", predict_ip(55, 1, 1))
    print("   3 ports, 60s gap, clean      ->", predict_ip(3, 60, 0))
    print("  15 ports, 3s gap, clean       ->", predict_ip(15, 3, 0))
    print("   2 ports, 50s gap, clean      ->", predict_ip(2, 50, 0))


if __name__ == "__main__":
    main()
