from flask import Flask, render_template, request
import os
import replicate
from transformers import AutoTokenizer  # Import AutoTokenizer here

app = Flask(__name__)

# Set Replicate API token
os.environ["REPLICATE_API_TOKEN"] = "r8_1ZABppSsnyYqzEz3FCy9J2BBgZc4Ncp2cqPmc"

# Initialize chatbot components
from llama_index.llms import Replicate
from llama_index import set_global_tokenizer
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index import ServiceContext
from llama_index import VectorStoreIndex, SimpleDirectoryReader

llm, index = None, None
api = replicate.Client(api_token=os.environ["REPLICATE_API_TOKEN"])  # Move the Replicate API client creation here

from transformers import AutoTokenizer

def initialize_chatbot():
    global llm, index, api
    llama2_7b_chat = "meta/llama-2-7b-chat:8e6975e5ed6174911a6ff3d60540dfd4844201974602551e10e9e87ab143d81e"
    llm = Replicate(
        model=llama2_7b_chat,
        temperature=0.01,
        additional_kwargs={"top_p": 1, "max_new_tokens": 300},
    )

    tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf")
    set_global_tokenizer(tokenizer)

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    service_context = ServiceContext.from_defaults(
        llm=llm, embed_model=embed_model
    )

    documents = SimpleDirectoryReader("./data").load_data()
    index = VectorStoreIndex.from_documents(
        documents, service_context=service_context
    )


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form['user_input']
    bot_response = chat(user_input)
    return render_template('index.html', user_input=user_input, bot_response=bot_response)

def chat(user_input):
    prompt = user_input
    output = api.run(
        "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
        input={"prompt": prompt}
    )

    response = ""
    for item in output:
        response += item

    return response

if __name__ == '__main__':
    initialize_chatbot()
    app.run(debug=True)
