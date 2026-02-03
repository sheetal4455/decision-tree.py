import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import matplotlib.pyplot as plt

# Load dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Split data
X = df[iris.feature_names]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Decision Tree
clf = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# Accuracy
accuracy = clf.score(X_test, y_test)
print(f"\nAccuracy on test data: {accuracy*100:.2f}%")

# Print tree structure
tree_rules = export_text(clf, feature_names=iris.feature_names)
print("\nDecision Tree Structure:\n")
print(tree_rules)

# Plot tree
plt.figure(figsize=(12,8))
plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=True)
plt.show()

# Predict a new sample
new_sample = [[5, 3, 1.5, 0.2]]
prediction = clf.predict(new_sample)
print(f"\nPredicted class for new sample {new_sample[0]}: {iris.target_names[prediction[0]]}")
