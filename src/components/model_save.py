#!/usr/bin/env python
# coding: utf-8

# Import Required Libraries

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_ckpt = "google/pegasus-cnn_dailymail"

tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt).to(device)

gen_kwargs = {"length_penalty": 0.8, "num_beams":8, "max_length": 128}

pipe = pipeline("summarization", model=model_pegasus,tokenizer=tokenizer)
            
logging.info('Model saved in artifacts')
         

except Exception as e:
    logging.info('Exception occured at Model Saving')
    raise CustomException(e, sys)
