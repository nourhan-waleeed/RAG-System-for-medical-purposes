from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os
import uvicorn


app = FastAPI()

# gemini api key
genai.configure(api_key="AIzaSyAtJvMN3wnVInIq3Wb16S-dfAJQcEU-2eE")

# Initialize FLAN-T5 model
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Initialize ChromaDB
# CHROMA_DATA_PATH = f"C:\Users\nourh\OneDrive\Desktop\odoo\cleopatra\chromadb_storage"
client = chromadb.PersistentClient(path='chromadb_storage')
print(client)
collection = client.get_collection(name="rag_collection")
qa_cache = client.get_collection(name="cache")


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str


def get_emb(text):
    model = "models/embedding-001"
    embedding = genai.embed_content(
        model=model,
        content=text,
        task_type="retrieval_document"
    )
    return embedding['embedding']


def get_relevant_docs(user_query):
    query_embeddings = get_emb(user_query)
    results = collection.query(
        query_embeddings=[query_embeddings],
        n_results=3
    )
    return [results['documents'][0][i] for i in range(len(results['ids'][0]))]


def make_rag_prompt(query, relevant_passage):
    relevant_passage = ''.join(relevant_passage)
    prompt = (
        f"You are a helpful and informative chatbot that answers questions using text from the reference passage included below. "
        f"Respond in a complete sentence and make sure that your response is easy to understand for everyone. "
        f"Maintain a friendly and conversational tone. If the passage is irrelevant, feel free to ignore it.\n\n"
        f"QUESTION: '{query}'\n"
        f"PASSAGE: '{relevant_passage}'\n\n"
        f"ANSWER:"
    )
    return prompt


def generate_response(user_prompt):
    model = genai.GenerativeModel('gemini-pro')
    answer = model.generate_content(user_prompt)
    return answer.text


def adjust_answer_with_flan_t5(query, previous_answer, max_length=150):
    prompt = f"Adjust this answer: '{previous_answer}' to better fit this question: '{query}'. Provide a concise and relevant response."
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def cache_qa_pair(query, answer, query_embedding):
    import time
    cache_id = f"qa_{int(time.time())}"
    qa_cache.upsert(
        ids=[cache_id],
        documents=[answer],
        embeddings=[query_embedding],
        metadatas=[{"original_query": query}]
    )


def generate_answer(query):
    query_embedding = get_emb(query)
    results = qa_cache.query(
        query_embeddings=[query_embedding],
        n_results=1
    )

    if results['distances'][0] and results['distances'][0][0] < 0.92:
        previous_answer = results['documents'][0][0]
        adjusted_answer = adjust_answer_with_flan_t5(query, previous_answer)
        return adjusted_answer
    else:
        relevant_text = get_relevant_docs(query)
        prompt = make_rag_prompt(query, relevant_text)
        answer = generate_response(prompt)
        cache_qa_pair(query, answer, query_embedding)
        return answer


@app.get("/")
def read_root():
    return {"status": "healthy", "message": "RAG System API is running"}


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    # try:
    answer = generate_answer(request.question)
    return AnswerResponse(answer=answer)
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8999)
