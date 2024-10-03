from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from model import update_resume_content, extract_text_from_file  # Import AI and text extraction functions

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'resume' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['resume']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        resume_text = extract_text_from_file(filepath)
        return jsonify({'message': 'File uploaded successfully', 'resume_text': resume_text}), 200
    return jsonify({'error': 'Invalid file format'}), 400

@app.route('/update_resume', methods=['POST'])
def update_resume():
    data = request.json
    job_description = data.get('job_description')
    resume_text = data.get('resume_text')

    if not job_description or not resume_text:
        return jsonify({'error': 'Job description or resume text missing'}), 400
    
    try:
        updated_resume_text = update_resume_content(resume_text, job_description)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    return jsonify({'updated_resume_text': updated_resume_text}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
