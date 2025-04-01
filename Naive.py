# Step 1: Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Load Dataset
df = pd.read_csv('Naive_dataset.csv')  # Ensure the dataset contains 'text' and 'label' columns
print(df.head())  # Display first few rows

# Step 3: Preprocess Data
df['text'] = df['text'].str.replace('[^a-zA-Z]', ' ', regex=True).str.lower()  # Clean text

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Step 4: Convert Text to Features
vectorizer = CountVectorizer()  # Convert text to numerical features
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 5: Train the Naive Bayes Classifier
nb_classifier = MultinomialNB()  # Initialize classifier
nb_classifier.fit(X_train_vec, y_train)  # Train the model

# Step 6: Evaluate the Model
y_pred = nb_classifier.predict(X_test_vec)  # Predict on test data

# Print performance metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
