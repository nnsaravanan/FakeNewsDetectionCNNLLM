import os, csv, json, time
from datetime import datetime
import pandas as pd
import numpy as np
import google.generativeai as genai
import logging

# Configure logging

# get api and required data

with open("keys/googleapi", 'r') as f:
    gemini_key = f.read().strip()

df = pd.read_csv("data/test6000.csv")
texts = df['text'].to_numpy()
labels = df['label'].to_numpy()

'''with open("code/text.txt", 'r') as f:
    text = f.read().strip()'''

# set up the gemini prompt stuff
genai.configure(api_key = gemini_key)
model = genai.GenerativeModel('gemini-pro')

prompt = 'Summarize this text into 2 sentences and then provide a basic :'
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

# set up mapping and conf matrix for checking tp, fp, tn, fn
mapping = {'fake':0,
           'real':1}

conf = {'tp':0,
        'tn':0,
        'fp':0,
        'fn':0}

def log_request_submission(ct):
    logging.info(f"Request {ct} submitted")

# Logging statements for request completion
def log_request_completion(ct):
    logging.info(f"Request {ct} finished")

# Logging statements for request failure
def log_request_failure(ct, error):
    logging.error(f"Request {ct} failed with an error: {error}")

fails = dict()


if __name__ == "__main__":
    current_datetime = datetime.now()

    # Format the date and time as mm-dd-yy-hh-min
    folder_name = current_datetime.strftime("%m-%d-%y-%H-%M")

    # Create the folder path
    folder_path = os.path.join("results", folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    logging.basicConfig(filename=os.path.join(folder_path,'app.log'), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    for ct, values in enumerate(zip(texts, labels), start=1):  # Start counting from 1
        text, label = values
        log_request_submission(ct)  # Log request submission
        try:
            response = model.generate_content(f"{prompt} {text}", safety_settings=safe)
            response = mapping[response.text.lower()]
            

        except Exception as e:
            log_request_failure(ct, e)  # Log request failure
            fails[text] = str(e)

        finally:
            log_request_completion(ct)  # Log request completion
            with open(os.path.join(folder_path,'conf.json'), "w") as f: 
                json.dump(conf, f)
            with open(os.path.join(folder_path,"fails.json"), "w") as f: 
                json.dump(fails, f)
            time.sleep(1)

    logging.info("All texts submitted")
    logging.info("program finished")