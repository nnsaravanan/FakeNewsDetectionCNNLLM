from Gemini import GeminiAPI

class PromptLoader:

    def __init__(self, model: GeminiAPI.Model) -> None:

        self.model = model
    

    def send_prompt(self, statement: str) -> str:
        
        response = self.model.send_prompt(statement)
        return response.text


    def check_true(self, factors: list, statement:str) -> bool:

        string_to_send = f"Determine if this statement is true based on these factors {factors}. Respond with only true or false: {statement}"
        response = self.model.send_prompt(string_to_send)

        if response.text.lower()=="false":
            return False
        return True
    

    def rephrase(self, requirements: dict, statement: str) -> str:

        string_to_send = f"rewrite this sentence with these requirements {requirements}: {statement}"
        response = self.model.send_prompt(string_to_send)
        return response.text

    
    def summarize(self, features: list, statement: str) -> str:

        string_to_send = f"Summarize this statement and include these in the summary {features} : {statement}"
        response = self.model.send_prompt(string_to_send)
        return response.text
    