# app/main.py
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import google.generativeai as genai
import os
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)  # Fixed: __name__ instead of _name_

# Load environment variables
load_dotenv()

# Load API key securely
GENAI_API_KEY = os.getenv("GENAI_API_KEY")
if not GENAI_API_KEY:
    logger.error("GENAI_API_KEY is missing!")
    raise ValueError("GENAI_API_KEY is missing! Set it in your .env file or Render environment.")

# Load datasets
diabetes_faq_path = "app/diabetes_faq.csv"
app_faq_path = "app/app_faq.csv"

if not os.path.exists(diabetes_faq_path) or not os.path.exists(app_faq_path):
    logger.error("One or both FAQ datasets are missing!")
    raise FileNotFoundError("One or both FAQ datasets are missing!")

diabetes_df = pd.read_csv(diabetes_faq_path)
app_df = pd.read_csv(app_faq_path)
df = pd.concat([diabetes_df, app_df], ignore_index=True)

# Initialize FastAPI
app = FastAPI()

# Initialize Google Gemini API
genai.configure(api_key=GENAI_API_KEY)

# Load Sentence Transformer Model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode all dataset questions
dataset_questions = df["question"].astype(str).tolist()
dataset_answers = df["answer"].astype(str).tolist()
dataset_embeddings = model.encode(dataset_questions, convert_to_tensor=True)

# Define request model
class QuestionRequest(BaseModel):
    question: str

def find_similar_answer(user_question, threshold=0.80):
    user_embedding = model.encode(user_question, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(user_embedding, dataset_embeddings)[0]
    best_match_idx = scores.argmax().item()
    best_score = scores[best_match_idx].item()
    if best_score >= threshold:
        return dataset_answers[best_match_idx]
    return None

def get_gemini_response(question, max_words=60):
    try:
        model = genai.GenerativeModel("gemini-1.5-pro-latest")
        prompt = f"Answer the following question in {max_words} words or less: {question}"
        response = model.generate_content(prompt)
        return response.text.strip() if response.text else "Sorry, I couldn't generate a response."
    except Exception as e:
        logger.error(f"Gemini API error: {str(e)}")
        return f"Error fetching response from Gemini API: {str(e)}"

@app.get("/")
def read_root():
    return {"message": "FastAPI chatbot is running!"}

@app.post("/chat/")
async def chat(request: QuestionRequest):
    user_question = request.question.strip()
    answer = find_similar_answer(user_question)
    if answer:
        return {"response": answer}
    gemini_response = get_gemini_response(user_question)
    return {"response": gemini_response}

if __name__== "__main__":
    port = int(os.getenv("PORT", 10000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)
