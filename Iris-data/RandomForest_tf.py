import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# Read the data from a CSV file
data_path = "C:/Users/User/OneDrive/Desktop/Kaggle/iris/iris.csv"
column_names = ["feature1", "feature2", "feature3", "feature4", "class"]
data = pd.read_csv(data_path, header=None, names=column_names)
print(data.head())

# Split the data into features and labels
features = data.drop("class", axis=1)
labels = data["class"]

# Split the dataset into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.3, random_state=42)

# Create the input function for TensorFlow
def input_fn(features, labels, training=True, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    if training:
        dataset = dataset.shuffle(1000).repeat()
    return dataset.batch(batch_size)

# Define the feature columns
feature_columns = []
for feature_name in features.columns:
    feature_columns.append(tf.feature_column.numeric_column(feature_name))

# Create the Random Forest classifier
classifier = RandomForestClassifier(n_estimators=100)

# Train the classifier
classifier.fit(train_features, train_labels)

# Make predictions on the testing set
predictions = classifier.predict(test_features)

# Calculate accuracy, F1 score, precision, and recall
accuracy = classifier.score(test_features, test_labels)
classification_report = classification_report(test_labels, predictions)
confusion_mtx = confusion_matrix(test_labels, predictions)

# Print the evaluation metrics
print("Test set accuracy: {accuracy:0.3f}".format(accuracy=accuracy))
print("Classification Report:")
print(classification_report)
print("Confusion Matrix")
print(confusion_mtx)
print("Confusion Matrix with predicted and golden labels")
print(pd.DataFrame(confusion_mtx, index= labels.unique(), columns=labels.unique()))

