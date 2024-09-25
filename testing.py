import cv2
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load the model
with open('model_new.p', 'rb') as f:
    model_dict = pickle.load(f)
model = model_dict['model']

# Set up video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess the frame for model input
    # Assuming you have the same preprocessing as during training
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Perform landmark detection here if needed and prepare data for model
    # Example: data = preprocess_image(img_rgb)

    # Make prediction
    # prediction = model.predict(np.array([data]))

    # Display the result on the frame
    # label = np.argmax(prediction, axis=1)
    # cv2.putText(frame, f"Sign: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show the frame
    cv2.imshow('Sign Language Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
