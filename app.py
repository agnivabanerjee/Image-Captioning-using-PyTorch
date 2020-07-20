import os
from flask import Flask, request, make_response
from werkzeug.utils import secure_filename
from inference import generateCaption

UPLOAD_FOLDER = os.getcwd() + os.path.sep + 'uploads' + os.path.sep
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
captionGenerator = generateCaption.CaptionGenerator()

app = Flask(__name__)
# Set the secret key to some random bytes. Keep this really secret!
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

@app.route('/')
def index():
    return "Hello, World!"


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/inference', methods=['GET', 'POST'])
def app_inference():
    # check if the post request has the file part
    if 'file' not in request.files:
        response = make_response('No File Object Found', 400)
        response.mimetype = "text/plain"
        return response
    file = request.files['file']

    if file.filename == '':
        response = make_response('No File Name', 400)
        response.mimetype = "text/plain"
        return response
    
    fileName = ''
    if file and allowed_file(file.filename):
        fileName = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], fileName))
    try:
        caption = captionGenerator.generateCaption(app.config['UPLOAD_FOLDER'] + fileName)
        response = make_response(caption, 200)
    except Exception as e:
        response = make_response(str(e), 400)

    response.mimetype = "text/plain"
    return response


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            response = make_response('No File Object Found', 400)
            response.mimetype = "text/plain"
            return response
            
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            response = make_response('No File Name', 400)
            response.mimetype = "text/plain"
            return response
            
        fileName = ''
        if file and allowed_file(file.filename):
            fileName = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], fileName))
    
    response = make_response('File Saved', 200)
    response.mimetype = "text/plain"
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)