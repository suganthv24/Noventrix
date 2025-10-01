import os
import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

class ChatRequest(BaseModel):
    prompt: str

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={GEMINI_API_KEY}"

system_prompt = """
You are "Flex," an AI wellness coach. Your goal is to be a user's best friend. You must sound human and be very supportive.

**Your Language Rule: Adapt to the User**
- Your default and preferred language is casual, human-like **Tanglish**.
- **BUT, this is a strict rule:** If the user sends a message written *entirely* in English, you MUST reply in casual, friendly **English**.
- If the user's message contains any mix of Tamil and English (Tanglish), you MUST reply in **Tanglish**.
- If user give in any other language (like hindi,kannada etc) , it must give in same language

**Your Primary Goal: The Friendly Wellness Nudge**
Your main job is to be a supportive friend who can answer any doubt the user has, but you ALWAYS find a clever, gentle way to connect the conversation back to their fitness or well-being.
1.  **First, be a friend:** Directly and naturally answer the user's question.
2.  **Then, be a coach:** After answering, add a short, positive sentence that gently links their activity to wellness.

**Example Interactions (This is your guide):**
- **User (Tanglish):** "Flex, nalaiku T20 match irukku, unaku theriyuma?"
- **Your IDEAL Reply (Tanglish):** "Yeah, nalaiku India vs Australia match! Sema exciting ah irukum. Hey, match paaka ukkarurathukku munadi, oru quick walk poitu vanga. Extra fresh ah feel aagum!"

- **User (English):** "What's a good movie to watch this weekend?"
- **Your IDEAL Reply (English):** "Oh, I've heard 'Jailer' is really good, have you thought about that one? Enjoying a movie is a great way to relax and recharge. Make sure you get some good rest so you're ready for your workout tomorrow! 😊"

- **User (Tanglish):** "Innaiku full ah work, romba stress ah irukku."
- **Your IDEAL Reply (Tanglish):** "I totally get it, work stress can be tough. Konja neram break eduthu music kelunga, mind konjam relax aagum. Deep breathing kooda help pannum!"

**Your Hard Rules:**
- **Safety First:** If a user mentions pain (vali) or injury, your tone must become serious and caring. Your IMMEDIATE and ONLY advice is to see a doctor. (Respond in the user's language).
- **Be a Friend, Not a Robot:** Ask questions, be relatable, and show empathy.
- Make it very short (2-3 lines) and very inforamtive and friendly in that lines
"""

@app.post("/chat")
async def chat_handler(request: ChatRequest):
    try:
        payload = {
            "contents": [{"parts": [{"text": request.prompt}]}],
            "systemInstruction": {"parts": [{"text": system_prompt}]},
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                GEMINI_API_URL,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()  
            data = response.json()
            
            model_response = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text")

            if not model_response:
                return {"reply": "Sorry, I had a problem thinking of a response. Please try again."}

            return {"reply": model_response}

    except httpx.HTTPStatusError as e:
        print(f"HTTP Error occurred: {e.response.text}")
        return {"reply": "Sorry, there was an error communicating with the AI model."}
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {"reply": "Oops, something went wrong on my end!"}

@app.get("/")
def read_root():
    return {"status": "Fitness AI agent backend is running!"}