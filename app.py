from flask import Flask, request, render_template
from flask_cors import CORS
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

app = Flask(__name__)
CORS(app)

model_name = "facebook/blenderbot-400M-distill"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
conversation_history = []

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/chatbot', methods=['POST'])
def handle_prompt():
    global conversation_history
    
    data = request.get_json()
    input_text = data['prompt']

    # Add the user input to the conversation history
    conversation_history.append(tokenizer.eos_token + input_text)

    # Create conversation history string
    history = " ".join(conversation_history)

    # Tokenize the input text and history
    inputs = tokenizer.encode(history, return_tensors="pt")

    # Generate the response from the model
    outputs = model.generate(inputs, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Add the model response to the conversation history
    conversation_history.append(tokenizer.eos_token + response)

    # return json.dumps({"response": response})
    return response

if __name__ == '__main__':
    app.run(debug=True)