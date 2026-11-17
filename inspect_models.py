import joblib

FILES = {
    "ANOMALY": "models/anomaly_model.joblib",
    "CONDITION": "models/condition_model.joblib",
    "RUL": "models/rul_model.joblib",
}


for name, path in FILES.items():

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    obj = joblib.load(path)

    print("Main object type:", type(obj))

    if isinstance(obj, dict):

        print("\nDictionary keys:")

        for key in obj:
            value = obj[key]

            print(f"\n[{key}]")
            print("  Type:", type(value))

            if hasattr(value, "n_features_in_"):
                print(
                    "  Number of features:",
                    value.n_features_in_
                )

            if hasattr(value, "feature_names_in_"):
                print(
                    "  Feature names:",
                    list(value.feature_names_in_)
                )

    else:

        if hasattr(obj, "n_features_in_"):
            print(
                "Number of features:",
                obj.n_features_in_
            )

        if hasattr(obj, "feature_names_in_"):
            print(
                "Feature names:",
                list(obj.feature_names_in_)
            )