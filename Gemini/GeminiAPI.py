import google.generativeai as genai

class Model:

    def __init__(self, api_key: str) -> None:

        self.api_key = api_key
        self.safety = safe = [
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
        
        genai.configure(api_key = api_key)
        self.model = genai.GenerativeModel('gemini-pro')
    
    def send_prompt(self, prompt: str) -> str:

        response = self.model.generate_content(f"{prompt}", safety_settings=self.safety)
        return response