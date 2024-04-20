import random
import json
import torch
from model import NeuralNet
from nltk_utils import bow, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

with open("intents.json", 'r') as f:
    intents = json.load(f)

FILE = 'data.pth'
data = torch.load(FILE)

input_size = data['input_size']
hidden_size = data['hidden_size']
output_size = data['output_size']
tags = data['tags']
all_words = data['all_words']
model_state = data['model_state']

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = 'DAV Helper'
# print("Hi there!! I am DAV helper, how can I help you? ")
# print("Type exit to quit")

def get_response(sentence):
    sentence = tokenize(sentence)
    X = bow(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim = 1)
    tag = tags[predicted.item()]

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["intent"]:
                return f"{random.choice(intent['responses'])}"
    else:
        return f"I do not understand..."