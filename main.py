# import uvicorn
# from fastapi import FastAPI
# from pydantic import BaseModel
# import pandas as pd
# import google.generativeai as genai
# from sentence_transformers import SentenceTransformer, util

# #  Load the dataset
# df = pd.read_csv("diabetes_faq.csv")

# #  Initialize FastAPI
# app = FastAPI()

# #  Initialize Google Gemini API
# GENAI_API_KEY = "AIzaSyDG7cVaGVzmeH78ztkYGkDdI8PpFPYDRco"  # 🔥 Replace with your API key
# genai.configure(api_key=GENAI_API_KEY)

# #  Load Sentence Transformer Model for better matching
# model = SentenceTransformer("all-MiniLM-L6-v2")

# #  Encode all dataset questions
# dataset_questions = df["question"].astype(str).tolist()
# dataset_answers = df["answer"].astype(str).tolist()
# dataset_embeddings = model.encode(dataset_questions, convert_to_tensor=True)

# #  Define request model
# class QuestionRequest(BaseModel):
#     question: str  # Expecting "question" in JSON

# def find_similar_answer(user_question, threshold=0.7):
#     """
#     Find the most similar question in the dataset using Sentence Transformers.
#     """
#     user_embedding = model.encode(user_question, convert_to_tensor=True)
#     scores = util.pytorch_cos_sim(user_embedding, dataset_embeddings)[0]
    
#     best_match_idx = scores.argmax().item()
#     best_score = scores[best_match_idx].item()

#     if best_score >= threshold:  # ✅ Only return if similarity is high
#         return dataset_answers[best_match_idx]
#     return None  # No match found

# def get_gemini_response(question):
#     """
#     Generate a response using Gemini AI if no match is found in the dataset.
#     """
#     try:
#         model = genai.GenerativeModel("gemini-1.5-pro-latest")
#         response = model.generate_content(question)
#         return response.text.strip()  # Clean response

#     except Exception as e:
#         return f"Error fetching response from Gemini API: {str(e)}"

# @app.post("/chat/")
# async def chat(request: QuestionRequest):
#     user_question = request.question.strip()

#     #  Step 1: Find similar question using Sentence Transformers
#     answer = find_similar_answer(user_question)
#     if answer:
#         return {"source": "dataset", "response": answer}

#     #  Step 2: If no match, get response from Gemini API
#     gemini_response = get_gemini_response(user_question)
#     return {"source": "gemini", "response": gemini_response}


# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000) 



import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, util
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load API key securely
GENAI_API_KEY = os.getenv("GENAI_API_KEY")
#  Load the dataset
df = pd.read_csv("diabetes_faq.csv")

#  Initialize FastAPI
app = FastAPI()

#  Initialize Google Gemini API
genai.configure(api_key=GENAI_API_KEY)

#  Load Sentence Transformer Model for better matching
model = SentenceTransformer("all-MiniLM-L6-v2")

#  Encode all dataset questions
dataset_questions = df["question"].astype(str).tolist()
dataset_answers = df["answer"].astype(str).tolist()
dataset_embeddings = model.encode(dataset_questions, convert_to_tensor=True)

#  Define request model
class QuestionRequest(BaseModel):
    question: str  # Expecting "question" in JSON

def find_similar_answer(user_question, threshold=0.7):
    """
    Find the most similar question in the dataset using Sentence Transformers.
    """
    user_embedding = model.encode(user_question, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(user_embedding, dataset_embeddings)[0]
    
    best_match_idx = scores.argmax().item()
    best_score = scores[best_match_idx].item()

    if best_score >= threshold:  # only return if similarity is high
        return dataset_answers[best_match_idx]
    return None  # No match found
def get_gemini_response(question, max_words=60):
    """
    Generate a concise response using Gemini AI.
    """
    try:
        model = genai.GenerativeModel("gemini-1.5-pro-latest")
        prompt = f"Answer the following question in {max_words} words or less: {question}"
        response = model.generate_content(prompt)
        return response.text.strip()  # Corrected position of return
    except Exception as e:
        return f"Error fetching response from Gemini API: {str(e)}"


# def get_gemini_response(question, max_words=60):
#     """
#     Generate a concise response using Gemini AI.
#     """
#     try:
#         model = genai.GenerativeModel("gemini-1.5-pro-latest")
#         prompt = f"Answer the following question in {max_words} words or less: {question}"
#     except Exception as e:
#         response = model.generate_content(prompt)
#         return response.text.strip()

#         return f"Error fetching response from Gemini API: {str(e)}"

@app.get("/")
def read_root():
    return {"message": "FastAPI is running!"}

@app.post("/chat/")
async def chat(request: QuestionRequest):
    user_question = request.question.strip()

    #  Step 1: Find similar question using Sentence Transformers
    answer = find_similar_answer(user_question)
    if answer:
        return {"source": "dataset", "response": answer}

    #  Step 2: If no match, get response from Gemini API
    gemini_response = get_gemini_response(user_question)
    return {"source": "gemini", "response": gemini_response}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))  # Use Render's PORT or default to 8000
    uvicorn.run(app, host="0.0.0.0", port=port)




# import uvicorn
# from fastapi import FastAPI
# from pydantic import BaseModel
# import pandas as pd
# import google.generativeai as genai
# from sentence_transformers import SentenceTransformer, util
# import os
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# # Load API key securely
# GENAI_API_KEY = os.getenv("GENAI_API_KEY")
# if not GENAI_API_KEY:
#     raise ValueError("Missing GENAI_API_KEY. Please set it in your environment variables.")

# # Initialize Google Gemini API
# genai.configure(api_key=GENAI_API_KEY)

# # Load dataset with a safe file path
# DATASET_PATH = os.path.join(os.getcwd(), "diabetes_faq.csv")

# if not os.path.exists(DATASET_PATH):
#     raise FileNotFoundError(f"Dataset file not found: {DATASET_PATH}")

# df = pd.read_csv(DATASET_PATH)

# # Initialize FastAPI
# app = FastAPI()

# # Load Sentence Transformer Model for better matching
# model = SentenceTransformer("all-MiniLM-L6-v2")

# # Encode all dataset questions
# dataset_questions = df["question"].astype(str).tolist()
# dataset_answers = df["answer"].astype(str).tolist()
# dataset_embeddings = model.encode(dataset_questions, convert_to_tensor=True)

# # Define request model
# class QuestionRequest(BaseModel):
#     question: str  # Expecting "question" in JSON

# def find_similar_answer(user_question, threshold=0.7):
#     """
#     Find the most similar question in the dataset using Sentence Transformers.
#     """
#     user_embedding = model.encode(user_question, convert_to_tensor=True)
#     scores = util.pytorch_cos_sim(user_embedding, dataset_embeddings)[0]

#     best_match_idx = scores.argmax().item()
#     best_score = scores[best_match_idx].item()

#     if best_score >= threshold:  # only return if similarity is high
#         return dataset_answers[best_match_idx]
#     return None  # No match found

# def get_gemini_response(question, max_words=60):
#     """
#     Generate a concise response using Gemini AI.
#     """
#     try:
#         model = genai.GenerativeModel("gemini-1.5-pro-latest")
#         prompt = f"Answer the following question in {max_words} words or less: {question}"
#         response = model.generate_content(prompt)
#         return response.text.strip()
#     except Exception as e:
#         return f"Error fetching response from Gemini API: {str(e)}"

# @app.get("/")
# def read_root():
#     return {"message": "FastAPI is running!"}

# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 8000))  # Use Render's PORT or default to 8000
#     print(f"Starting server on port {port}...")  # Debugging log
#     uvicorn.run(app, host="0.0.0.0", port=port)

# @app.post("/chat/")
# async def chat(request: QuestionRequest):
#     user_question = request.question.strip()

#     # Step 1: Find similar question using Sentence Transformers
#     answer = find_similar_answer(user_question)
#     if answer:
#         return {"source": "dataset", "response": answer}

#     # Step 2: If no match, get response from Gemini API
#     gemini_response = get_gemini_response(user_question)
#     return {"source": "gemini", "response": gemini_response}





# import os
# import json
# import pandas as pd
# from pathlib import Path
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import google.generativeai as genai
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()
# GENAI_API_KEY = os.getenv("GENAI_API_KEY")
# PORT = int(os.getenv("PORT", 8000))  # Use Railway's PORT variable

# if not GENAI_API_KEY:
#     raise ValueError("GENAI_API_KEY is missing. Please set it in Railway's environment variables.")

# # Initialize Google Generative AI model
# try:
#     genai.configure(api_key=GENAI_API_KEY)
#     gemini_model = genai.GenerativeModel("gemini-1.5-pro-latest")
# except Exception as e:
#     raise RuntimeError(f"Failed to initialize Gemini AI: {str(e)}")

# # Define dataset path
# DATASET_PATH = Path(__file__).parent / "diabetes_faq.csv"

# # Load dataset (handle missing file gracefully)
# faq_data = None
# if DATASET_PATH.exists():
#     faq_data = pd.read_csv(DATASET_PATH)
#     faq_data.dropna(subset=["Question", "Answer"], inplace=True)
# else:
#     print(f"Warning: Dataset file '{DATASET_PATH}' not found. Proceeding without it.")

# # FastAPI app instance
# app = FastAPI()

# # Pydantic model for API requests
# class QuestionRequest(BaseModel):
#     question: str
#     max_words: int = 60

# # Function to get AI-generated response
# def get_gemini_response(question: str, max_words: int = 60):
#     try:
#         prompt = f"Answer the following question in {max_words} words or less: {question}"
#         response = gemini_model.generate_content(prompt)
#         return response.text.strip()
#     except Exception as e:
#         return f"Error fetching response from Gemini API: {str(e)}"

# # Route: Health Check
# @app.get("/health")
# def health_check():
#     return {"status": "healthy"}

# # Route: Get answer from dataset or AI
# @app.post("/get_answer")
# def get_answer(request: QuestionRequest):
#     user_question = request.question.lower()

#     # Check if dataset exists and find the answer
#     if faq_data is not None:
#         for _, row in faq_data.iterrows():
#             if user_question in row["Question"].lower():
#                 return {"answer": row["Answer"]}

#     # If not found, use AI model
#     ai_response = get_gemini_response(user_question, request.max_words)
#     return {"answer": ai_response}

# # Run the app on Railway
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=PORT)
