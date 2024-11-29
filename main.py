import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

generator = pipeline('text-generation', model='EleutherAI/gpt-neo-125M')
print(generator("Hello, world!", max_length=50))



from src.qa_agent import query_gpt4_turbo
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

question = "What is artificial intelligence?"
context = "Artificial intelligence (AI) refers to the simulation of human intelligence in machines."

response = query_gpt4_turbo(question, context)
print("Response:", response)