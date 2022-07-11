import pandas as pd
from pathlib import Path

from flask import Flask
from flask import render_template, request, redirect, url_for, flash
from dog_breed_algo import dog_breed_algorithm


UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['SECRET_KEY'] = 'the random string'


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
@app.route('/index')
def index():
    return render_template('master.html')


# web page that handles user query and displays model results
@app.route('/classify', methods=['GET', 'POST'])
def classify():
    if request.method == "POST":
        if request.files:
            # Get image object
            image = request.files["filename"]

            if not allowed_file(image.filename):
                flash('No file part')
                return redirect(request.url)

            # Save image and close
            filename = image.filename
            image_path = Path(app.config['UPLOAD_FOLDER']) / filename
            image.save(image_path)
            image.close()

            # Run dog breed algorithm
            prediction = dog_breed_algorithm(str(image_path))

            return render_template("classify_result.html", filename=filename, prediction=prediction)

    # This will render the go.html Please see that file.
    return redirect(url_for('index'))


@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()
