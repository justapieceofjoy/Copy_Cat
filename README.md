# copyCat - Futuristic Plagiarism Checker

Copy Cat is a plagiarism detection system that uses semantic and character-level comparison techniques to detect copied or paraphrased text. It features a sleek, neon-themed web interface built using Flask templates and standard HTML/CSS. The backend leverages modern NLP tools such as Sentence Transformers, spaCy, and Levenshtein distance to provide detailed similarity metrics and highlight suspected plagiarized segments.

## Features

- **Multiple Input Methods:**
  - **Upload Files:** Users can upload two documents.
  - **Paste Text:** Alternatively, users can paste text directly into provided text areas.

- **Plagiarism Detection:**
  - **Document-Level Analysis:**
    - Computes cosine similarity using Sentence Transformers.
    - Uses Levenshtein distance for character-level similarity.
    - Combines both metrics to produce an overall similarity percentage.
  - **Sentence-Level Analysis:**
    - Splits documents into sentences.
    - Computes a similarity matrix and averages maximum sentence similarity.
  - **Highlighting:**
    - Highlights matching tokens in Document 1.
    - Adjusts the minimum match length based on overall similarity (more aggressive highlighting when similarity is high).
  - **Heatmap Visualization:**
    - Generates a heatmap of sentence-level similarities for visual inspection.

- **Futuristic, Responsive UI:**
  - Built with Flask template engine using HTML and CSS.
  - Dark, neon-themed design with purple accents.
  - Clean, minimalistic, and responsive interface optimized for both desktop and mobile.

## Project Structure

```
copyCat/
├── app.py                  # Flask backend application
├── requirements.txt        # Python dependencies
└── templates/
    ├── index.html          # Homepage for file upload / text paste
    └── results.html        # Results page displaying similarity metrics, highlighted text, and heatmap
testData/
├── document1.txt           # Test Document 1 to check
└── document2.txt           # Test Document 2 to check from
```

## Getting Started

### Prerequisites

- **Python 3.7+ and pip:**  
  Download from [python.org](https://www.python.org/).

### Setting Up the Project

0. **Set up from Google Drive**

  - Download the copyCat folder and open it in a Code Editor.
  - Then run the app.py in Code Editor or in the terminal run:

   ```bash
   python app.py
   ```

  - Open your web browser and navigate to `http://127.0.0.1:5000`
  - Upload two desired files, input your desired text or paste text from or upload testData/document1 and testData/document2.

1. **Clone or Download the Project:**

   ```bash
   git clone https://github.com/yourusername/copyCat.git
   cd copyCat
   ```

2. **Create and Activate a Virtual Environment (Optional but Recommended):**

   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Python Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   _Your `requirements.txt` should include packages like:_
   ```
   Flask
   nltk
   spacy
   sentence-transformers
   python-Levenshtein
   matplotlib
   ```

4. **Download Necessary NLP Resources:**

   ```bash
   python -m nltk.downloader punkt
   python -m spacy download en_core_web_sm
   ```

### Running the Application

1. **Start the Flask Application:**

   ```bash
   python app.py
   ```

   The server will run by default on `http://127.0.0.1:5000`.

2. **Access the Application:**

   Open your web browser and navigate to `http://127.0.0.1:5000`. You can then:
   - Upload two files or paste text for testData/document1 and testData/document2.
   - Submit the form to run plagiarism detection.
   - View the results with similarity percentages, highlighted text, and a heatmap visualization.

## How It Works

- **Backend Processing:**
  - The Flask backend reads input documents (either file upload or pasted text).
  - It preprocesses the text using advanced techniques (normalization, lemmatization, stopword removal with spaCy).
  - It computes document-level similarity using Sentence Transformers and Levenshtein ratio.
  - It also performs sentence-level analysis by splitting texts into sentences and comparing them.
  - The backend highlights matching tokens in Document 1 based on the overall similarity, adjusting the minimum match length dynamically.
  - A heatmap is generated to visualize sentence-level similarity.

- **User Interface:**
  - The application uses simple HTML templates (`index.html` and `results.html`) with inline CSS.
  - The interface features a futuristic, neon-inspired design using purple accents.
  - The results page displays detailed similarity metrics, highlighted text (using `<span>` tags with a highlight class), and an image of the heatmap.
  
## Customization and Future Improvements

- **Enhanced Highlighting:**  
  Adjust or refine the highlighting logic to capture more nuanced cases of paraphrasing.
- **Advanced NLP Models:**  
  Experiment with more sophisticated transformer models or additional similarity metrics.
- **UI/UX Enhancements:**  
  Consider integrating modern front-end frameworks or libraries if further customization is desired.
- **Performance Optimizations:**  
  Improve scalability by caching embeddings or employing asynchronous processing.


## Acknowledgments

- **Flask** for providing the lightweight backend framework.
- **spaCy, Sentence Transformers, and Levenshtein** libraries for robust text processing and similarity computation.
