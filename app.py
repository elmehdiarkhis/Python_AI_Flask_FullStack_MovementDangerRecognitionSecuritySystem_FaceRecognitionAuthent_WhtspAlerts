#app.py
from flask import Flask, render_template, request, send_from_directory, Response
from movement_recognition import process_frame, load_dangerous_faces
from movement_recognition import authenticate_user
from movement_recognition import save_user_face
import cv2
import os
import time
from twilio.rest import Client
import threading
import json
from flask import session, redirect, url_for

app = Flask(__name__, static_folder='.')


#TWILIO WHATSAPP---------------------
account_sid = 'ACc7bfce1ece15da8af70319a882a6cdd6'
auth_token = 'e0a9aa1bae30d66b326f54e802a83aff'
client = Client(account_sid, auth_token)
#------------------------------------


#================================================================================================
def send_whatsapp_message(to, body):
    message = client.messages.create(
        body=body,
        from_='whatsapp:+14155238886',
        to=f'whatsapp:{to}'
    )
    print(f"Message sent: {message.sid}")

danger_detected_images_path = "danger_detected_images"
os.makedirs(danger_detected_images_path, exist_ok=True)

def save_danger_image(frame):
    timestamp = time.strftime("%Y_%m_%dll%H:%M:%S")
    file_name = f"{danger_detected_images_path}/frame_{timestamp}.jpg"
    cv2.imwrite(file_name, frame)
    return file_name

mouvement_detected_images_path = "mouvement_detected_images"
os.makedirs(mouvement_detected_images_path, exist_ok=True)

def save_image(frame):
    timestamp = time.strftime("%Y_%m_%dll%H:%M:%S")
    file_name = f"{mouvement_detected_images_path}/frame_{timestamp}.jpg"
    cv2.imwrite(file_name, frame)
    return file_name

dangerous_faces_encodings, dangerous_faces_names = load_dangerous_faces("dangerous_faces_images")
cam = cv2.VideoCapture(0)

def video_loop():
    while not stop_loop:
        success, frame1 = cam.read()
        success, frame2 = cam.read()
        processed_frame, movement_detected, danger_detected, detected_dangerous_person_name = process_frame(frame1, frame2, dangerous_faces_encodings, dangerous_faces_names)

        if movement_detected:
            save_image(frame1)
            if danger_detected:
                saved_image_path = save_danger_image(frame1)
                send_whatsapp_message("+14382238023", f"A dangerous person has been detected! with the name of {detected_dangerous_person_name}")

def generate():
    while True:
        success, frame = cam.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
#================================================================================================




# Add this function to clear session data before the first request
@app.before_first_request
def clear_session():
    session.clear()

# Change the index route to check if the user is logged in
@app.route('/')
def index():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

# Add the login route
@app.route('/login')
def login():
    return render_template('login.html')

# Add the login_check route
@app.route('/login_check')
def login_check():
    success, _ = authenticate_user(cam)
    if success:
        session['logged_in'] = True
        return 'success'
    else:
        return 'failed'



@app.route('/start')
def start():
    global stop_loop
    stop_loop = False
    t = threading.Thread(target=video_loop)
    t.start()
    return "Video loop started"

@app.route('/stop')
def stop():
    global stop_loop
    stop_loop = True
    return "Video loop stopped"

@app.route('/get_saved_images')
def get_saved_images():
    file_list = sorted(os.listdir(mouvement_detected_images_path))
    image_paths = [f"/mouvement_detected_images/{file}" for file in file_list]
    return json.dumps(image_paths)

@app.route('/mouvement_detected_images/<path:path>')
def send_image(path):
    return send_from_directory('mouvement_detected_images', path)

@app.route('/get_danger_saved_images')
def get_danger_saved_images():
    file_list = sorted(os.listdir(danger_detected_images_path))
    image_paths = [f"/danger_detected_images/{file}" for file in file_list]
    return json.dumps(image_paths)

@app.route('/danger_detected_images/<path:path>')
def send_danger_image(path):
    return send_from_directory('danger_detected_images', path)

@app.route('/video_feed')
def video_feed():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/signup')
def signup():
    return render_template('signup.html')


@app.route('/signup_check')
def signup_check():
    success, message = save_user_face(cam, "authorized_faces_images")
    if success:
        return 'success'
    else:
        return message



if __name__ == '__main__':
    # Add this line to the end of the file, right before `app.run(...)`
    app.secret_key = 'supersecretkey'
    app.run(debug=True, host='0.0.0.0', port=8000)


