import google.generativeai as genai

class Model:
    def __init__(self, api_key: str) -> None:
        """
        Initializes the Model object.

        Parameters:
        - api_key (str): The API key for accessing the generative AI service.
        """
        self.api_key = api_key
        self.safety = [
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
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
    
    def send_prompt(self, prompt: str) -> str:
        """
        Sends a prompt to the generative AI model and receives a response.

        Parameters:
        - prompt (str): The prompt provided to the model.

        Returns:
        - response (str): The generated content response from the model.
        """
        response = self.model.generate_content(f"{prompt}", safety_settings=self.safety)
        return response
