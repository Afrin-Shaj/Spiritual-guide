import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai

load_dotenv()

# Configure the Gemini API key
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Define the FastAPI app
app = FastAPI()

# Allow CORS for your MERN application
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your frontend's origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the request body model
class UserInput(BaseModel):
    input_text: str

# Create the generation configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Create the model
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction="You are a spirit guide. When the user shares their emotions, provide quotes based on their current mood. Make them feel good. Use holy books or any other author's short speech. If they ask for 'how' questions, reply like a guide for them. Make their mental health stable...",
)

# Define the chatbot endpoint
@app.post("/chat")
async def chat(user_input: UserInput):
    try:
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(user_input.input_text)
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the app (uncomment the following line to run the server directly)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=9000)  # Changed to 'localhost'
