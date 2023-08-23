from flask import Flask, render_template, send_from_directory
import os
import face_recognition
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

app = Flask(__name__)

id_card_dir = 'content\Ids'
user_faces_dir = 'content/faces'

# Manually select the images for comparison
id_card_images = [
    'id1.jpg', 'id3.jpg', 'id4.jpg', 'id5.jpg', 'id6.jpg', 
    'id1.jpg', 'id1.jpg', 'id1.jpg', 'id1.jpg'
]

user_face_images = [
    'f1.jpg', 'f3.jpg', 'f4.jpg', 'f5.jpg', 'f6.jpg',
    'f3.jpg', 'f4.jpg', 'f5.jpg', 'f6.jpg'
]

id_card_encodings = []
user_face_encodings = []

for id_card_image in id_card_images:
    image = face_recognition.load_image_file(os.path.join(id_card_dir, id_card_image))
    face_locations = face_recognition.face_locations(image)
    
    if len(face_locations) > 0:
        encoding = face_recognition.face_encodings(image, [face_locations[0]])[0]
        id_card_encodings.append(encoding)
    else:
        print(f"No face found in ID card image: {id_card_image}")

for user_face_image in user_face_images:
    image = face_recognition.load_image_file(os.path.join(user_faces_dir, user_face_image))
    face_locations = face_recognition.face_locations(image)
    
    if len(face_locations) > 0:
        encoding = face_recognition.face_encodings(image, [face_locations[0]])[0]
        user_face_encodings.append(encoding)
    else:
        print(f"No face found in user face image: {user_face_image}")

if len(id_card_encodings) > 0 and len(user_face_encodings) > 0:
    true_labels = []
    predicted_labels = []

    results = []

    comparisons = [
        (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3)
    ]

    for id_idx, user_idx in comparisons:
        true_labels.extend(['ID Card', 'User Face'])
        
        similarity = face_recognition.face_distance([user_face_encodings[user_idx]], id_card_encodings[id_idx])
        match = similarity < 0.6
        results.append((id_card_images[id_idx], user_face_images[user_idx], 'Match' if match else 'No Match'))
        predicted_labels.extend(['ID Card' if match else 'User Face', 'User Face'])

    confusion_mat = confusion_matrix(true_labels, predicted_labels, labels=['ID Card', 'User Face'])

    # Save confusion matrix image
    fig, ax = plt.subplots(figsize=(6, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=['ID Card', 'User Face'])
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    plt.savefig('static/images/confusion_matrix.png')
    
    # Construct a results DataFrame
    results_df = pd.DataFrame(results, columns=['ID Card', 'User Face', 'Result'])

@app.route('/')
def index():
    return render_template('index.html', confusion_matrix='static/images/confusion_matrix.png', results=results_df.to_html(index=False))

@app.route('/static/images/<path:filename>')
def download_file(filename):
    return send_from_directory('static/images', filename)

if __name__ == '__main__':
    app.run(debug=True)
