# AI Resume–Job Matcher

An AI-powered Resume–Job Matcher that analyzes a resume against a job description and provides skill matching, missing skills, semantic similarity, and AI-generated career suggestions.

## Features

- Upload resume in PDF format
- Extract resume text automatically
- Identify relevant technical skills
- Compare resume skills with job requirements
- Find matching skills
- Identify missing skills
- Calculate semantic similarity between resume and job description
- Generate AI-powered resume and career analysis using an LLM
- Streamlit web interface

## Technologies Used

- Python
- Streamlit
- PyPDF
- Sentence Transformers
- Scikit-learn
- Hugging Face Transformers
- PyTorch
- Qwen LLM
- NLP
- Semantic Similarity
- Cosine Similarity

## How It Works

```text
Resume PDF
    ↓
Text Extraction
    ↓
Text Cleaning
    ↓
Skill Extraction
    ↓
Resume vs Job Skill Comparison
    ↓
Sentence Embeddings
    ↓
Cosine Similarity
    ↓
Semantic Match Score
    ↓
LLM Analysis
    ↓
Career Recommendations

====================================================================

Installation:

Clone the repository:

git clone https://github.com/YOUR_USERNAME/AI-Resume-Job-Matcher.git

Go to the project directory:

cd AI-Resume-Job-Matcher

Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run app.py

=================================================================

Output:

The application provides:

Resume skills
Job-required skills
Matching skills
Missing skills
Semantic match score
AI-generated resume analysis
Skill improvement suggestions

====================================================================

Future Improvements:

Resume scoring based on job requirements
Experience and education matching
ATS compatibility analysis
Better skill extraction using NLP models
Resume improvement recommendations
Support for multiple job descriptions
Cloud deployment