import os
from flask import Flask, render_template, request
from model import analyze_skin

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files.get('photo')
    if not file:
        return "No file uploaded", 400

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(save_path)

    result = analyze_skin(save_path)
    return render_template('result.html', image=file.filename, diagnosis=result)

if __name__ == '__main__':
    app.run(debug=True)
