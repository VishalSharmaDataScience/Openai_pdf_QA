import pdfplumber
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util
import os


# Load a pre-trained sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file.
    :param pdf_path: Path to the PDF file.
    :return: Extracted text as a single string.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from PDF: {e}")

def extract_keywords(questions):
    """
    Extract keywords from a list of questions.
    :param questions: List of user-supplied questions.
    :return: Set of keywords.
    """
    if not questions:
        raise ValueError("Questions cannot be empty")

    try:
        stop_words = set(stopwords.words("english"))
    except:
        raise RuntimeError("Failed to load stop words")

    keywords = set()

    for question in questions:
        print("Processing question:", question)
        if not question:
            raise ValueError("Question cannot be empty")

        try:
            words = word_tokenize(question)
            filtered_words = [word for word in words if word.lower() not in stop_words]
            keywords.update(filtered_words)
        except Exception as e:
            raise RuntimeError(f"Failed to extract keywords: {e}")

    return keywords

def chunk_pdf_by_keywords_and_semantics(text, keywords, question, max_tokens=1500):
    """
    Chunk the PDF text into relevant sections based on keywords and re-rank by semantic similarity.
    :param text: Extracted text from the PDF.
    :param keywords: Keywords derived from user questions.
    :param question: User's question for semantic ranking.
    :param max_tokens: Maximum tokens per chunk (GPT-4-friendly).
    :return: List of top-ranked chunks.
    """
    # Step 1: Chunk text by sentences
    sentences = text.split(". ")  # Split text into sentences.
    chunks = []
    current_chunk = ""

    # Group sentences into chunks containing keywords
    for sentence in sentences:
        if any(keyword.lower() in sentence.lower() for keyword in keywords):
            if len(current_chunk.split()) + len(sentence.split()) > max_tokens * 0.75:  # Chunk limit
                chunks.append(current_chunk.strip())
                current_chunk = ""
            current_chunk += sentence + ". "


    # Append the last chunk if it's non-empty
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # Handle edge case: No matching chunks
    if not chunks:
        print("No chunks matched keywords; splitting text into general chunks.")
        chunks = [text[i : i + max_tokens * 4] for i in range(0, len(text), max_tokens * 4)]

    # Step 2: Remove empty chunks
    chunks = [chunk for chunk in chunks if chunk.strip()]
    if not chunks:
        raise ValueError("No valid chunks could be generated from the PDF text.")

    # Step 3: Re-rank chunks by semantic similarity
    try:
        question_embedding = model.encode(question, convert_to_tensor=True)
        chunk_embeddings = model.encode(chunks, convert_to_tensor=True)
        print("Question embedding:", len(question_embedding))
        similarities = util.cos_sim(question_embedding, chunk_embeddings)[0]
        print("Similarities:", similarities)
        ranked_chunks = sorted(zip(similarities, chunks), key=lambda x: x[0], reverse=True)
        
        # Return top-ranked chunks (limit to top 3 for efficiency)
        chunk_list = [chunk for _, chunk in ranked_chunks[:3]]
        
        return chunk_list
    except Exception as e:
        print(f"Error during semantic similarity computation: {e}")
        return []
