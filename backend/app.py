from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from retriever import retrieve_context
import os
import requests
import json

app = FastAPI()

# Frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

    if not query or not act_name:
        raise HTTPException(status_code=400, detail="Missing 'question' or 'act_name' in request.")

    # 1. Retrieve context
    try:
        # Assuming retrieve_context returns a string
        context = retrieve_context(query, act_name) 
    except Exception as e:
        # Raise 500 error if retrieval fails (e.g., file not found, bad embeddings)
        raise HTTPException(status_code=500, detail=f"Error during context retrieval: {str(e)}")

    # 2. Construct prompt
    prompt = f"Answer based on the following legal text: {context} \nQuestion: {query}"

    # 3. Call Hugging Face Inference API
    # ... (headers, payload setup as before) ...

    response = requests.post(
        f"https://api-inference.huggingface.co/models/{HF_MODEL}", 
        data=json.dumps(payload)
    )

    if response.status_code != 200:
        # Raise 502 Bad Gateway if the external service (HF) fails
        raise HTTPException(
            status_code=502, 
            detail=f"Hugging Face API failed: {response.status_code}. Response: {response.text}"
        )

    # 4. Return successful answer
    try:
        answer = response.json()[0]["generated_text"]
        return {"answer": answer}
    except (IndexError, KeyError) as e:
        # Handle unexpected JSON response structure from the API
        raise HTTPException(status_code=502, detail=f"Failed to parse answer from Hugging Face response: {str(e)}")

# Cross-platform startup
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
