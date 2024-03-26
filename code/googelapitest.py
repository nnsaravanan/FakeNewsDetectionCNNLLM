import os
import google.generativeai as genai


with open("keys/googleapi", 'r') as f:
    gemini_key = f.read().strip()

genai.configure(api_key = gemini_key)
model = genai.GenerativeModel('gemini-pro')

for i in range(5):
    response = model.generate_content(f"Please give me the answer to 10 + {i}")

    print(response.text)