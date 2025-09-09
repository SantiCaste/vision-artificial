import pandas as pd
import dataset_generator as dg
from sklearn import tree
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    # Load dataset and preprocess
    dataset = pd.read_csv(dg.HU_MOMENTS_FILE, header=None)

    # Separate features and labels
    y = dataset.iloc[:, 0].values
    X = dataset.iloc[:, 1:].values

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Training the model
    classifier = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=4, min_samples_leaf=2)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    # Evaluate the model
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

    # view the model
    tree.plot_tree(classifier)

    # Save the model
    dump(classifier, 'shape_classifier.joblib')