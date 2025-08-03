# train.py
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from preprocess import preprocess_data

# Load and preprocess data
X, y, le_gender, le_device, le_location = preprocess_data('data/Indian_Kids_Screen_Time.csv')

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model and encoders
joblib.dump(model, 'model/screen_time_classifier.pkl')
joblib.dump(le_gender, 'model/le_gender.pkl')
joblib.dump(le_device, 'model/le_device.pkl')
joblib.dump(le_location, 'model/le_location.pkl')
