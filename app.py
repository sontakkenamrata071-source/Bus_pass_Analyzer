from flask import Flask, render_template, request, jsonify, url_for, flash, session, redirect
import psycopg2
import os
import cv2
import numpy as np
import base64
import time
import winsound
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__, template_folder='template')
app.secret_key = "supersecretkey"

# PostgreSQL Database Configuration
DB_SETTINGS = {
    "dbname": "busPass",
    "user": "postgres",
    "password": "namu",
    "host": "localhost",
    "port": "5432",
}

#Helper to get connection to database
def get_db_connection():
    return psycopg2.connect(**DB_SETTINGS)

# Directory to store captured images
CAPTURED_IMAGES_DIR = "captured_images"  
os.makedirs(CAPTURED_IMAGES_DIR, exist_ok=True)


# Predefined admin credentials
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "1234"


# Home Page
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/template/admin/login.html", methods=["GET", "POST"])
def admin_login():
    error_message = None
    if request.method == "POST":
        # Retrieve form data
        username = request.form.get("username")
        password = request.form.get("password")

        # Validate credentials
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session["admin_logged_in"] = True  # Set login state in session
            return redirect("/template/admin/dashboard.html")  # Redirect to dashboard
        else:
           error_message = "Invalid credentials. Please try again."
           
    # Render login page for GET request
    return render_template("admin/login.html", error_message=error_message)

@app.route("/template/admin/dashboard.html", methods=["GET", "POST"])
def admin_dashboard():
    if not session.get("admin_logged_in"):
        flash("You must log in first.", "warning")
        return redirect("/template/admin/login.html")  # Redirect to login if not logged in
    current_date = datetime.now().date()
    try:
        # Fetch users for display
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, full_name, contact_number, address, arrival_stop, destination_stop, bus_pass_validity FROM student_registration")
        users = cursor.fetchall()
        # print(users)  #debugging
        cursor.close()
        conn.close()
    except Exception as e:
        flash(f"Error fetching users: {e}", "danger")
        users = []

    return render_template("admin/dashboard.html", users=users, current_date=current_date)

@app.route("/admin/users/add", methods=["POST"])
def add_user():
    try:
        # Retrieve form data
        full_name = request.form.get("fullName")
        contact_number = request.form.get("contactNumber")
        address = request.form.get("address")
        arrival_stop = request.form.get("arrivalStop")
        destination_stop = request.form.get("destinationStop")
        bus_pass_validity = request.form.get("busPassValidity")

        conn = get_db_connection()
        cursor = conn.cursor()
        query = """
        INSERT INTO student_registration (full_name, contact_number, address, arrival_stop, destination_stop, bus_pass_validity)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        cursor.execute(query, (full_name, contact_number, address, arrival_stop, destination_stop, bus_pass_validity))
        conn.commit()
        cursor.close()
        conn.close()
        flash("User added successfully.", "success")
    except Exception as e:
        flash(f"Error adding user: {e}", "danger")
    return redirect("/template/admin/dashboard.html")

@app.route("/admin/users/update/<int:user_id>", methods=["POST"])
def update_user(user_id):
    try:
        # Retrieve updated data
        full_name = request.form.get("fullName")
        contact_number = request.form.get("contactNumber") 
        address = request.form.get("address")
        arrival_stop = request.form.get("arrivalStop")
        destination_stop = request.form.get("destinationStop")
        bus_pass_validity = request.form.get("busPassValidity")

        # Update the user in the database
        conn = get_db_connection()
        cursor = conn.cursor()
        query = """
        UPDATE student_registration
        SET full_name = %s, contact_number = %s, address = %s, arrival_stop = %s, destination_stop = %s, bus_pass_validity = %s
        WHERE id = %s
        """
        cursor.execute(query, (full_name, contact_number, address, arrival_stop, destination_stop, bus_pass_validity, user_id))
        conn.commit()
        cursor.close()
        conn.close()
        flash("User updated successfully.", "success")
    except Exception as e:
        flash(f"Error updating user: {e}", "danger")
    return redirect("/template/admin/dashboard.html")

@app.route("/admin/users/delete/<int:user_id>", methods=["POST"])
def delete_user(user_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM student_registration WHERE id = %s", (user_id,))
        conn.commit()
        cursor.close()
        conn.close()
        flash("User deleted successfully.", "success")
    except Exception as e:
        flash(f"Error deleting user: {e}", "danger")
    return redirect("/template/admin/dashboard.html")

@app.route("/logout")
def logout():
    session.pop("admin_logged_in", None)
    flash("Logged out successfully.", "info")
    return render_template("/template/admin/login.html")


@app.route("/template/student/register.html")
def student_register():
    return render_template("student/register.html")

# Load pre-trained face cascade for detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_image(image):
    """
    Preprocess the image for face recognition by converting to grayscale and resizing.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (100, 100))
    gray = cv2.equalizeHist(gray)
    return gray

def compute_cosine_similarity(vector1, vector2):
    return cosine_similarity([vector1], [vector2])[0][0]

def train_recognizer():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces_data = []
    labels = []
    label_to_name_map = {}
    current_label = 0

    # Loop through each folder in the captured_images directory
    for folder_name in os.listdir(CAPTURED_IMAGES_DIR):
        folder_path = os.path.join(CAPTURED_IMAGES_DIR, folder_name)
        if not os.path.isdir(folder_path):
            continue

        # Use folder name as user name
        user_name = folder_name.replace("_", " ").strip()
        label_to_name_map[current_label] = user_name
        print(f"Processing folder: {user_name}")

        for img_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_file)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                print(f"Skipping invalid image: {img_path}")
                continue

            image = cv2.resize(image, (100, 100))
            image = cv2.equalizeHist(image)

            faces_data.append(image)
            labels.append(current_label)

        current_label += 1

    if not faces_data or not labels:
        raise ValueError("No valid faces found for training.")

    print("Training recognizer with loaded images...")
    recognizer.train(faces_data, np.array(labels))
    print("Recognizer training completed successfully.")
    return recognizer, label_to_name_map

@app.route("/template/student/verification.html", methods=['GET', 'POST'])
def student_verification():
    if request.method == 'GET':
        return render_template("student/verification.html")

    if request.method == 'POST':
        try:
            # Check if the request has JSON content
            if request.content_type != 'application/json':
                return jsonify({"message": "Unsupported Media Type. Use 'application/json'."}), 415

            # Get the JSON data from the request
            data = request.get_json()
            image_data = data.get('image')

            if not image_data:
                return jsonify({"message": "No image data provided."}), 400

            # Decode the base64 image
            encoded_data = image_data.split(',')[1]
            nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Convert to grayscale for better face detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(50, 50)
            )

            if len(faces) == 0:
                return jsonify({'verified': False, 'message': 'No face detected. Please try again with proper lighting.'}), 400

            # Load and train the recognizer
            recognizer, label_to_name_map = train_recognizer()

            for (x, y, w, h) in faces:
                captured_face = gray[y:y + h, x:x + w]
                captured_face = cv2.resize(captured_face, (100, 100))
                captured_face = cv2.equalizeHist(captured_face)

                # Predict the face
                label, confidence = recognizer.predict(captured_face)
                print(f"Predicted Label: {label}, Confidence: {confidence}")

                if confidence > 80:
                    winsound.Beep(1000, 500)
                    return jsonify({'verified': False, 'message': 'Face not recognized. Please try again.'})

                # Get the user name from the label map
                user_name = label_to_name_map.get(label).strip()
                print(f"Predicted User: '{user_name}'")
                print(f"Querying database for: '{user_name}'")

                # Check bus pass validity in the database
                conn = get_db_connection()
                cursor = conn.cursor()

                cursor.execute("""SELECT bus_pass_validity FROM student_registration WHERE TRIM(full_name) = TRIM(%s)""", (user_name,))
                result = cursor.fetchone()
                cursor.close()
                conn.close()

                if result:
                    bus_pass_validity = result[0]
                    today = datetime.now().date()

                    if bus_pass_validity > today:
                        return jsonify({
                            'verified': True,
                            'message': f"Hi {user_name}, your bus pass is valid until {bus_pass_validity}."
                        })
                    else:
                        winsound.Beep(1000, 500)
                        return jsonify({
                            'verified': False,
                            'message': f"Hi {user_name}, your bus pass expired on {bus_pass_validity}. Please renew it."
                        })
                        
                        
                winsound.Beep(1000, 500)
                return jsonify({'verified': False, 'message': 'User data not found in the database. Register Yourself'})

            return jsonify({'verified': False, 'message': 'Verification failed.'})

        except Exception as e:
            print("Error:", e)
            return jsonify({'verified': False, 'message': str(e)}), 500

# Capture Images and Save User Data
@app.route('/capture', methods=['POST'])
def capture_images():
    try:
        # Get form data
        full_name = request.form.get("fullName")
        contact_number = request.form.get("contactNumber")
        address = request.form.get("address")
        arrival_stop = request.form.get("arrivalStop")
        destination_stop = request.form.get("destinationStop")
        bus_pass_validity = request.form.get("busPassValidity")
        payment_receipt = request.files.get("paymentReceipt")

        if not all([full_name, contact_number, address, arrival_stop, destination_stop, bus_pass_validity, payment_receipt]):
            return jsonify({"error": "Incomplete user data provided."}), 400

        # Save user data to the database
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO student_registration (full_name, contact_number, address, arrival_stop, destination_stop, bus_pass_validity, payment_receipt)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (full_name, contact_number, address, arrival_stop, destination_stop, bus_pass_validity, psycopg2.Binary(payment_receipt.read())))
        conn.commit()
        cursor.close()
        conn.close()

        # Create a folder for the user
        user_folder = os.path.join(CAPTURED_IMAGES_DIR, full_name)
        os.makedirs(user_folder, exist_ok=True)

        # Access webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return jsonify({"error": "Unable to access the webcam."}), 500

        # Instructions for capturing diverse images
        instructions = [
            "Look straight at the camera.",
            "Tilt your head slightly to the left.",
            "Tilt your head slightly to the right.",
            "Look up slightly.",
            "Look down slightly.",
            "Smile slightly.",
            "Keep a neutral expression.",
            "Look surprised.",
            "Close your eyes briefly.",
            "Blink twice."
        ]

        image_count = 0

        for instruction in instructions:
            print(f"Instruction: {instruction}")
            time.sleep(2)  # Allow time for the user to follow the instruction

            for _ in range(2):  # Capture 2 images for each instruction
                ret, frame = cap.read()
                if not ret:
                    continue

                # Convert to grayscale and preprocess
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

                for (x, y, w, h) in faces:
                    face = gray_frame[y:y + h, x:x + w]
                    resized_face = cv2.resize(face, (100, 100))
                    equalized_face = cv2.equalizeHist(resized_face)

                    # Save the captured face image
                    image_path = os.path.join(user_folder, f"image_{image_count + 1}.jpg")
                    cv2.imwrite(image_path, equalized_face)
                    image_count += 1
                    print(f"Captured image {image_count}: {image_path}")

                    if image_count >= 20:  # Stop after capturing 20 images
                        break

                if image_count >= 20:
                    break

            if image_count >= 20:
                break

        cap.release()

        return jsonify({"message": f"{image_count} images captured successfully for {full_name}."})

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500






if __name__ == "__main__":
    app.run(debug=True)


