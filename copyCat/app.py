import io
import re
import unicodedata
import base64
import difflib
import Levenshtein
import matplotlib.pyplot as plt
from flask import Flask, request, render_template
from sentence_transformers import SentenceTransformer, util
import nltk
import spacy
from nltk.tokenize import sent_tokenize
import os

# For debugging, print the current working directory:
print("Current Working Directory:", os.getcwd())

# Download required NLTK data (if not already done)
nltk.download('punkt')

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Create Flask app and explicitly set the templates folder.
app = Flask("copyCat", template_folder=os.path.join(os.path.dirname(__file__), "templates"))

def advanced_preprocess(text):
    # Normalize: convert to ASCII (remove accents/special characters)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    text = text.lower()
    # Remove any characters that are not letters or whitespace
    text = re.sub(r'[^a-z\s]', '', text)
    # Process text with spaCy
    doc = nlp(text)
    # Keep only alphabetic tokens, remove stopwords, and lemmatize
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens)

def split_into_sentences(text):
    return sent_tokenize(text)

def highlight_plagiarism(text1, text2, overall_similarity, default_min=3):
    """
    Highlights matching tokens in text1 that appear in text2.
    The minimum match length is adjusted based on overall_similarity:
      - If overall similarity is above 70%, a lower minimum (2 tokens) is used
        so that more short matches are highlighted.
      - Otherwise, the default (e.g., 3) is used.
    """
    # overall_similarity is expected as a percentage (0-100)
    min_match_length = 2 if overall_similarity > 70 else default_min
    tokens1 = text1.split()
    tokens2 = text2.split()
    matcher = difflib.SequenceMatcher(None, tokens1, tokens2)
    matching_blocks = matcher.get_matching_blocks()
    highlighted_tokens = []
    for i, token in enumerate(tokens1):
        highlight = False
        for (a, b, n) in matching_blocks:
            if n >= min_match_length and a <= i < a + n:
                highlight = True
                break
        if highlight:
            highlighted_tokens.append(f"<span class='highlight'>{token}</span>")
        else:
            highlighted_tokens.append(token)
    return " ".join(highlighted_tokens)

def plagiarism_detection(doc1_text, doc2_text, alpha=0.5):
    # Preprocess entire documents for document-level analysis.
    preprocessed_doc1 = advanced_preprocess(doc1_text)
    preprocessed_doc2 = advanced_preprocess(doc2_text)
    
    # Load Sentence Transformer model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
    # ---------------------------
    # Document-Level Comparison
    # ---------------------------
    doc_embedding1 = model.encode(preprocessed_doc1, convert_to_tensor=True)
    doc_embedding2 = model.encode(preprocessed_doc2, convert_to_tensor=True)
    doc_cos_sim = util.cos_sim(doc_embedding1, doc_embedding2).item()
    doc_lev_sim = Levenshtein.ratio(preprocessed_doc1, preprocessed_doc2)
    combined_doc_similarity = alpha * doc_cos_sim + (1 - alpha) * doc_lev_sim
    overall_similarity = combined_doc_similarity * 100  # as percentage
    
    # ---------------------------
    # Sentence-Level Comparison
    # ---------------------------
    sentences1 = split_into_sentences(doc1_text)
    sentences2 = split_into_sentences(doc2_text)
    preprocessed_sentences1 = [advanced_preprocess(s) for s in sentences1 if s.strip()]
    preprocessed_sentences2 = [advanced_preprocess(s) for s in sentences2 if s.strip()]
    sent_embeddings1 = model.encode(preprocessed_sentences1, convert_to_tensor=True)
    sent_embeddings2 = model.encode(preprocessed_sentences2, convert_to_tensor=True)
    similarity_matrix = util.cos_sim(sent_embeddings1, sent_embeddings2)
    max_similarities = [max(similarity_matrix[i]).item() for i in range(similarity_matrix.shape[0])]
    avg_sentence_similarity = (sum(max_similarities) / len(max_similarities)) if max_similarities else 0

    # ---------------------------
    # Highlight Similar Parts in Document 1
    # ---------------------------
    highlighted_text = highlight_plagiarism(doc1_text, doc2_text, overall_similarity)

    return {
        'doc_cos_sim': doc_cos_sim * 100,
        'doc_lev_sim': doc_lev_sim * 100,
        'combined_doc_similarity': overall_similarity,
        'avg_sentence_similarity': avg_sentence_similarity * 100,
        'similarity_matrix': similarity_matrix.cpu().numpy() if hasattr(similarity_matrix, 'cpu') else similarity_matrix,
        'highlighted_text': highlighted_text
    }

def generate_heatmap(similarity_matrix):
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(similarity_matrix, cmap='viridis')
    fig.colorbar(cax)
    ax.set_xlabel('Document 2 Sentences')
    ax.set_ylabel('Document 1 Sentences')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    heatmap_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return heatmap_base64

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        doc1_text = ""
        doc2_text = ""
        # If files are uploaded, use them; otherwise, use pasted text.
        if 'file1' in request.files and request.files['file1'].filename != "":
            doc1_text = request.files['file1'].read().decode('utf-8')
        elif request.form.get('text1'):
            doc1_text = request.form.get('text1')
        if 'file2' in request.files and request.files['file2'].filename != "":
            doc2_text = request.files['file2'].read().decode('utf-8')
        elif request.form.get('text2'):
            doc2_text = request.form.get('text2')
        if not doc1_text or not doc2_text:
            return "Both documents are required!", 400
        results = plagiarism_detection(doc1_text, doc2_text)
        heatmap_base64 = generate_heatmap(results['similarity_matrix'])
        return render_template('results.html', results=results, heatmap_base64=heatmap_base64)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
