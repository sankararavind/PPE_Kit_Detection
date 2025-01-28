from flask import Flask, render_template, Response, request, session
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from wtforms.validators import InputRequired
from werkzeug.utils import secure_filename
import os
import cv2

# Assuming YOLO_Video.py contains the video_detection function
try:
    from YOLO_Video import video_detection
except ImportError:
    print("Error: YOLO_Video.py not found. Please ensure it's in the same directory.")
    exit()  # Exit if the module is not found

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'  # Replace with a strong secret key
app.config['UPLOAD_FOLDER'] = 'static/files'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Run")

def generate_frames(path_x):
    try:
        yolo_output = video_detection(path_x)
        for detection_ in yolo_output:
            ret, buffer = cv2.imencode('.jpg', detection_)
            if not ret:
                continue # Skip if encoding fails
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    except Exception as e:
        print(f"Error processing video: {e}")
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + open('static/error.jpg', 'rb').read() + b'\r\n') #Return error image


def generate_frames_web():
    try:
        cap = cv2.VideoCapture(0)  # 0 for default webcam, change if needed
        if not cap.isOpened():
            raise IOError("Cannot open webcam")

        while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break
            yolo_output = video_detection(frame)
            for detection_ in yolo_output:
                ret, buffer = cv2.imencode('.jpg', detection_)
                if not ret:
                    continue
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        cap.release()
    except Exception as e:
        print(f"Error processing webcam: {e}")
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + open('static/error.jpg', 'rb').read() + b'\r\n') #Return error image

@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    session.clear()
    return render_template('indexproject.html')

@app.route("/webcam", methods=['GET', 'POST'])
def webcam():
    session.clear()
    return render_template('ui.html')

@app.route('/FrontPage', methods=['GET', 'POST'])
def front():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        session['video_path'] = filepath
        return render_template('videoprojectnew.html', form=form, uploaded=True) # Send confirmation
    return render_template('videoprojectnew.html', form=form, uploaded=False)

@app.route('/video')
def video():
    video_path = session.get('video_path')
    if video_path:
        return Response(generate_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "No video found.", 400  # Return an error if no video is in session

@app.route('/webapp')
def webapp():
    return Response(generate_frames_web(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)