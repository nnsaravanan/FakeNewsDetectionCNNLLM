import os
import google.generativeai as genai


with open("keys/googleapi", 'r') as f:
    gemini_key = f.read().strip()

with open("code/text", 'r') as f:
    text = f.read().strip()


genai.configure(api_key = gemini_key)
model = genai.GenerativeModel('gemini-pro')


safe = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]
response = model.generate_content("negate this sentence for me: Obama was a good person", safety_settings=safe)

print(response.text)