This documentation will guide you through the setup and usage of a Flask-based web application for face recognition using the face_recognition library. The application compares images from ID cards with user-provided face images and displays the results in the form of a confusion matrix and a results table.

#Prerequisites
Python 3.x installed on your system.
Basic knowledge of Flask and web development.
Setup
Clone the Repository: Start by cloning the GitHub repository containing the project:

bash
git clone https://github.com/your-username/face-recognition-web-app.git
cd face-recognition-web-app
Install Dependencies: Install the required Python packages using pip:

bash
pip install flask face_recognition matplotlib pandas scikit-learn


#Organize Image Directories:
Place the ID card images in the content/Ids directory and the user face images in the content/faces directory.

#Run the Application:
Run the Flask application by executing the app.py script:

python app.py

This will start the development server, and you'll see output indicating that the server is running.
Access the Web App:
Open a web browser and navigate to http://127.0.0.1:5000/ to access the face recognition web app.

#View Results:

The web app displays a confusion matrix and a results table on the home page.
Confusion Matrix: The confusion matrix shows the comparison between ID cards and user face images. The diagonal represents correct matches, and off-diagonal elements represent incorrect matches.
Results Table: The results table provides detailed information about each comparison, including the ID card image, user face image, and whether there is a match or not.
