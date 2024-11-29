import openai 
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access the OpenAI API key from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

def query_gpt4_turbo(question, context):
    """
    Query GPT-4 Turbo to answer a question based on context.
    :param question: User's question.
    :param context: Extracted PDF text.
    :return: Answer string.
    """
    try:
        print("Sending request to GPT-4 Turbo...")
        print("Question:", question)
        print("Context length:", len(context))
        print("Generated chunks:", context[:100])
        
        
    
        """# Call GPT-4 Turbo model
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Context:\n{context[:100]}\n\nQuestion:\n{question}"}
            ],
            temperature=0.7,  # Adjust for creativity
            max_tokens=300    # Limit response length
        )"""
        

        model_name = "EleutherAI/gpt-neo-125M" #"EleutherAI/gpt-neo-2.7B"  # Or use "EleutherAI/gpt-j-6B" for a larger model

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

   
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"

        response = generator(prompt, max_length=1000, num_return_sequences=1,truncation=True, do_sample=True, temperature=0.7)
        print(response)
        print(response[0]['generated_text'])



        
        return response.choices[0].message.content.strip()
    

    except Exception as e:
        print("Error with model:", e)
        # Optionally, switch to a backup model
       