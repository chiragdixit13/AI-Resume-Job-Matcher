import streamlit as st
import re
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

st.set_page_config(
    page_title="Resume Job Matcher",
    page_icon="📄"
)

st.title("📄 AI Resume–Job Matcher")

uploaded_file = st.file_uploader(
    "Upload your resume",
    type=["pdf"]
)

job_description = st.text_area(
    "Paste the job description"
)

skills = [
    "Python",
    "Machine Learning",
    "Deep Learning",
    "NLP",
    "Transformers",
    "BERT",
    "LLMs",
    "RAG",
    "FAISS",
    "ChromaDB",
    "TensorFlow",
    "PyTorch",
    "SQL",
    "Git",
    "GitHub",
    "Docker",
    "AWS"
]

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_llm():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32
    )

    model.eval()

    return tokenizer, model

model = load_embedding_model()

if uploaded_file and job_description:

    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    clean_text = re.sub(r'\s+', ' ', text).strip()

    resume_skills = [
        skill for skill in skills
        if skill.lower() in clean_text.lower()
    ]

    job_skills = [
        skill for skill in skills
        if skill.lower() in job_description.lower()
    ]

    matching_skills = [
        skill for skill in resume_skills
        if skill in job_skills
    ]

    missing_skills = [
        skill for skill in job_skills
        if skill not in resume_skills
    ]

    resume_embedding = model.encode(clean_text)
    job_embedding = model.encode(job_description)

    score = cosine_similarity(
        [resume_embedding],
        [job_embedding]
    )[0][0]

    st.subheader("Resume Skills")
    st.write(resume_skills)

    st.subheader("Job Skills")
    st.write(job_skills)

    st.subheader("Matching Skills")
    st.write(matching_skills)

    st.subheader("Missing Skills")
    st.write(missing_skills)

    st.subheader("Semantic Match Score")
    st.write(f"{score * 100:.2f}%")

    context = f"""
Resume:
{clean_text}

Job Description:
{job_description}

Resume Skills:
{resume_skills}

Job Required Skills:
{job_skills}

Matching Skills:
{matching_skills}

Missing Skills:
{missing_skills}

Semantic Match Score:
{score * 100:.2f}%
"""

    prompt = f"""
You are a career assistant.

Analyze the resume against the job description using only the information provided below.

Context:
{context}

Give:

1. Overall match analysis
2. Important missing skills
3. Specific improvement suggestions
4. Skills to prioritize learning

Keep the answer clear, practical and concise.
"""

    with st.spinner("Generating AI analysis..."):

        tokenizer, llm = load_llm()

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )

        with torch.no_grad():
            outputs = llm.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.7
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]

        answer = tokenizer.decode(
            new_tokens,
            skip_special_tokens=True
        )

    st.subheader("🤖 AI Resume Analysis")
    st.write(answer)