# FakeNewsDetectionNNLLM
Fake news detection by using Gemini API to augment text and a CNN to detect the fake news. The text is augmented through prompts which are sent to Gemini with style, tone, and negation specifications. The data is the run through a CNN to determine whether the news is fake or real. 

# How to use
Input data to the path /data/
Input glove.6B.100d.txt word embeddings to /data/
To use the gemini api an API key must be obtained and placed in /keys/googleapi

Please follow the code in createdata.ipynb and testcnn.ipynb to set up data and run the CNN. If the data just needs to be tested on the CNN then all the code can be copied from testcnn.ipynb.

