import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the data
with open('G:\\Project\\Sign_Language\\data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

# Inspect data shapes
for i, entry in enumerate(data_dict['data']):
    print(f"Entry {i} shape: {len(entry)}")

# Flatten the data if necessary
def flatten_data(data):
    # Find the length of the longest entry
    max_length = max(len(entry) for entry in data)
    # Pad or truncate each entry to the max length
    return np.array([entry + [0] * (max_length - len(entry)) for entry in data])

# Flatten the data to ensure consistency
data = flatten_data(data_dict['data'])
labels = np.array(data_dict['labels'])

# Split the data
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train the model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Predict and evaluate
y_predict = model.predict(x_test)
score = accuracy_score(y_test, y_predict)

print('{}% of samples were classified correctly!'.format(score * 100))

# Save the model
try:
    with open('model_new.p', 'wb') as f:
        pickle.dump({'model': model}, f)
except Exception as e:
    print(f"An error occurred while saving the model: {e}")
