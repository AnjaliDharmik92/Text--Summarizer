# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import torch

# Load the saved model
device = "cuda" if torch.cuda.is_available() else "cpu"

#Prediction

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

model_ckpt = "google/pegasus-cnn_dailymail"

tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt).to(device)

gen_kwargs = {"length_penalty": 0.8, "num_beams":8, "max_length": 128}

pipe = pipeline("summarization", model=model_pegasus,tokenizer=tokenizer)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
        
    if request.method == 'POST':
       
        text_to_summarize = request.form.get('text_to_summarize')

        my_prediction = pipe(News_Article, **gen_kwargs)[0]
              
        return render_template('result.html', summary = my_prediction)


if __name__ == '__main__':
    #app.run(debug=True)
    # start server with 81 port
    app.run(debug=True)
