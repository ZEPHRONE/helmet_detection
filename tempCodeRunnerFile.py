import cv2
import numpy as np
import tensorflow as tf

# Load the TensorFlow SavedModel
model = tf.saved_model.load(r'C:\Users\jyoti\OneDrive\Desktop\mpmc project2\model.savedmodel')

# Load the label mapping
with open(r'C:\Users\jyoti\OneDrive\Desktop\mpmc project2\labels\labels.txt', 'r') as file:
    labels = file.read().splitlines()

# Get the names of the output tensors
output_tensor_names = list(model.signatures["serving_default"].structured_outputs.keys())
print("Output tensor names:", output_tensor_names)

# Initialize video capture
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Preprocess the frame for the model
    input_tensor = tf.image.resize(frame, [224, 224])
    input_tensor = input_tensor / 255.0  # Normalize the pixel values
    input_tensor = tf.expand_dims(input_tensor, axis=0)  # Add batch dimension

    # Perform inference
    predictions = model.signatures["serving_default"](tf.constant(input_tensor))

    # Get the predicted class index
    class_index = np.argmax(predictions[output_tensor_names[0]].numpy())

    # Get the predicted label
    predicted_label = labels[class_index]

    # Display the result
    cv2.putText(frame, f"Prediction: {predicted_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Helmet Detection', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the window
cap.release()
cv2.destroyAllWindows()
