import streamlit as st
import cv2
import numpy as np
import tensorflow as tf

# Load TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path='logo-detection-model-v3-037.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load label map
with open('labelmap.txt', 'r') as f:
    labels = f.read().strip().split('\n')

# Define function for inferencing with TFLite model and displaying results

def tflite_detect_images(image, labels, min_conf=0.5):
    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    float_input = (input_details[0]['dtype'] == np.float32)
    input_mean = 127.5
    input_std = 127.5

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imH, imW, _ = image.shape
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

    if float_input:
        input_data = (np.float32(input_data) - input_mean) / input_std

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[1]['index'])[0]
    classes = interpreter.get_tensor(output_details[3]['index'])[0]
    scores = interpreter.get_tensor(output_details[0]['index'])[0]

    detections = []

    for i in range(len(scores)):
        if 0 < scores[i] <= 1.0 and scores[i] > min_conf:
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))

            object_name = labels[int(classes[i])]
            label = '%s: %d%%' % (object_name, int(scores[i] * 100))
            detections.append([object_name, scores[i], xmin, ymin, xmax, ymax])

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(image, (xmin, label_ymin - labelSize[1] - 10),
                          (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255), cv2.FILLED)
            cv2.putText(image, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    return image,detections

# Streamlit app
def main():
    st.title('SMGB Logo detection model')
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("")
        st.write("Detected results : ")

        # Perform object detection
        result_image, detections = tflite_detect_images(image,labels)
        # Display detection results as labels
        st.subheader("Detection Results:")
        for detection in detections:
            st.write(f"{detection[0]} - Confidence: {str(detection[1] * 100).split('.')[0]} %")

        # Display the image with detections
        st.image(result_image, caption="Object Detection Result", use_column_width=True)

if __name__ == '__main__':
    main()
