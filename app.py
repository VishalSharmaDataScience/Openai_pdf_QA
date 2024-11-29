import os
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from src.pdf_parser import extract_text_from_pdf, extract_keywords, chunk_pdf_by_keywords_and_semantics
from src.qa_agent import query_gpt4_turbo
from dotenv import load_dotenv
import json
import nltk
import os




# Add NLTK data path globally
nltk_data_dir = os.path.join(os.path.dirname(__file__), 'nltk_data')
nltk.data.path.append(nltk_data_dir)
nltk.download('punkt_tab')


# Load environment variables
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'data/input/'

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs("data/output", exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process_pdf():
    # Handle file upload
    if 'pdf' not in request.files:
        return render_template("index.html", error="No PDF file uploaded")

    pdf_file = request.files['pdf']
    if pdf_file.filename == '':
        return render_template("index.html", error="No selected file")

    # Save the uploaded PDF
    filename = secure_filename(pdf_file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    pdf_file.save(file_path)

    # Handle questions
    questions = request.form.get("questions")
    if not questions:
        return render_template("index.html", error="No questions provided")

    questions = [q.strip() for q in questions.split(",")]
    print(questions)

    # Extract text and keywords
    try:
        pdf_text = extract_text_from_pdf(file_path)
        keywords = extract_keywords(questions)
    except Exception as e:
        return render_template("index.html", error=f"Failed to process PDF: {e}")

    # Generate answers
    results = {}
    try:
        for question in questions:
            print("Processing question to generate answer:", question)
            chunks = chunk_pdf_by_keywords_and_semantics(pdf_text, keywords, question)
            print(len(chunks), "chunks generated")
            answers = query_gpt4_turbo(question, chunks[0])
            print("Response:",  answers)
            results[question] = answers if answers else "Data Not Available"
    except Exception as e:
        return render_template("index.html", error=f"Failed to generate answers: {e}")

    # Save results to file (optional)
    output_file = "data/output/answers.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    # Render results in the web app
    return render_template("index.html", results=results)

if __name__ == "__main__":
    app.run(debug=True)
