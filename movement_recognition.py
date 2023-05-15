#movement_recognition.py
import cv2
import face_recognition
import os
from os import listdir
import uuid


def authenticate_user(camera, authorized_faces_path="authorized_faces_images"):
    success, frame = camera.read()
    if not success:
        return False, "Failed to read camera"

    authorized_faces_encodings, authorized_faces_names = load_dangerous_faces(authorized_faces_path)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(frame_rgb)
    face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(authorized_faces_encodings, face_encoding)
        if True in matches:
            return True, "User authenticated"

    return False, "User not recognized"



def load_dangerous_faces(folder_path):
    images = []
    dangerous_faces_encodings = []
    dangerous_faces_names = []

    valid_image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = [f for f in listdir(folder_path) if os.path.splitext(f)[1].lower() in valid_image_extensions]

    for img_file in image_files:
        img = cv2.imread(f'{folder_path}/{img_file}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        face_encodings = face_recognition.face_encodings(img)
        if face_encodings:
            dangerous_faces_encodings.append(face_encodings[0])
            dangerous_faces_names.append(os.path.splitext(img_file)[0])

    return dangerous_faces_encodings, dangerous_faces_names

def process_frame(frame1, frame2, dangerous_faces_encodings, dangerous_faces_names):
    movement_detected = False
    danger_detected = False
    detected_dangerous_person_name = None

    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contours
    for c in contours:
        if cv2.contourArea(c) < 5000:
            continue
        movement_detected = True
        x, y, w, h = cv2.boundingRect(c)
        rect_color = (0, 255, 0)

        # Face recognition
        face_locations = face_recognition.face_locations(frame1)
        face_encodings = face_recognition.face_encodings(frame1, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(dangerous_faces_encodings, face_encoding)
            if True in matches:
                rect_color = (0, 0, 255)
                danger_detected = True
                index = matches.index(True)
                detected_dangerous_person_name = dangerous_faces_names[index]
                break

        cv2.rectangle(frame1, (x, y), (x+w, y+h), rect_color, 2)

    return frame1, movement_detected, danger_detected, detected_dangerous_person_name

def generate_unique_user_name(save_path):
    user_name = str(uuid.uuid4())
    while os.path.exists(os.path.join(save_path, f"{user_name}.jpg")):
        user_name = str(uuid.uuid4())
    return user_name

def save_user_face(camera, save_path):
    success, frame = camera.read()
    if not success:
        return False, "Failed to read camera"

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(frame_rgb)

    if not face_locations:
        return False, "No face detected"

    face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)
    if not face_encodings:
        return False, "Failed to encode face"

    authorized_faces_encodings, authorized_faces_names = load_dangerous_faces(save_path)
    for authorized_face_encoding in authorized_faces_encodings:
        matches = face_recognition.compare_faces([authorized_face_encoding], face_encodings[0])
        if True in matches:
            return False, "User already exists"

    user_name = generate_unique_user_name(save_path)
    cv2.imwrite(os.path.join(save_path, f"{user_name}.jpg"), frame)
    return True, "User face saved"



def main():
    cam = cv2.VideoCapture(0)

    dangerous_faces_encodings, dangerous_faces_names = load_dangerous_faces("dangerous_faces_images")

    while cam.isOpened():
        success, frame1 = cam.read()
        success, frame2 = cam.read()
        processed_frame, movement_detected, danger_detected, detected_dangerous_person_name = process_frame(frame1, frame2, dangerous_faces_encodings, dangerous_faces_names)
        cv2.imshow('Processed Frame', processed_frame)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
