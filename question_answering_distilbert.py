import torch
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering

# Load DistilBERT model and tokenizer
model_name = "distilbert-base-cased-distilled-squad"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForQuestionAnswering.from_pretrained(model_name)

# Creating the variables to store the context and question.
context = "The capital city of Nepal is Kathmandu."
question = "What is the capital city of Nepal?"

# Preprocessing the text using the tokenizer
inputs = tokenizer(context, question, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

answer_start = torch.argmax(outputs.start_logits)
answer_end = torch.argmax(outputs.end_logits)
answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end + 1]))

print("Question:", question)
print("Answer:", answer)