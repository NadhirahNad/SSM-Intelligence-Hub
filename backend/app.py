from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from retriever import retrieve_context
import os
import requests
import json

app = FastAPI()

# Allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with your frontend URL if needed
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

# Hugging Face API
HF_API_KEY = os.environ.get("HF_API_KEY")
HF_MODEL = "t5-small"  # Lightweight generation model

@app.get("/")
def root():
    return {"message": "SSM Intelligence Hub Backend is running!"}

@app.post("/ask")
async def ask_question(request: Request):
    data = await request.json()
    query = data.get("question")
    act_name = data.get("act_name")

    # Retrieve context from CSV + FAISS
    try:
        context = retrieve_context(query, act_name)
    except Exception as e:
        return {"answer": f"Error retrieving context: {str(e)}"}

    # Construct prompt for generation
    prompt = f"Answer based on law: {context} \nQuestion: {query}"

    # Call Hugging Face Inference API
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 150}}

    response = requests.post(
        f"https://api-inference.huggingface.co/models/{HF_MODEL}",
        headers=headers,
        data=json.dumps(payload)
    )

    if response.status_code != 200:
        return {"answer": f"Error from Hugging Face API: {response.text}"}

    answer = response.json()[0]["generated_text"]
    return {"answer": answer}

# Cross-platform startup
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
