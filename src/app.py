import os

from flask import Flask, render_template, Response

from main import predict_sign

BASE_DIR = os.path.dirname(__file__)
ROOT = os.path.dirname(BASE_DIR)
TEMPLATES_PATH = os.path.join(ROOT, 'templates')

app = Flask(__name__, template_folder=TEMPLATES_PATH)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(predict_sign(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
