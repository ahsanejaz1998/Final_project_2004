import joblib

def test_feature_consistency():
    # Load the vectorizer used during training
    vect_train = joblib.load('vectorizer.pkl')

    # Load the vectorizer from a new or testing setup
    vect_test = joblib.load('vectorizer_test.pkl')  # Assume a vectorizer for testing is provided

    # Get feature names
    features_train = set(vect_train.get_feature_names_out())
    features_test = set(vect_test.get_feature_names_out())

    # Check if feature sets match
    if features_train != features_test:
        raise ValueError("Feature sets do not match between training and testing vectorizers.")
    
    print("Feature consistency check passed.")

if __name__ == '__main__':
    test_feature_consistency()
