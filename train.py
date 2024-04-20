import json
from nltk_utils import tokenize, stem, bow
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import NeuralNet

with open("intents.json", 'r') as f:
    intents = json.load(f)

# print(intents)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent["intent"]
    tags.append(tag)

    for pattern in intent['text']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))
# print(all_words)
ignore_words = ['?', '!', '.', ',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words  = sorted(set(all_words))
tags  = sorted(set(tags))

X_train = []
Y_train = []
for (word_sequence, tag) in xy:
    bag = bow(word_sequence, all_words)
    X_train.append(bag)
    
    label = tags.index(tag)
    Y_train.append(label)

X_train = np.array(X_train)
Y_train = np.array(Y_train)


class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = Y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

batch_size = 8

input_size = len(all_words)
hidden_size = 8
output_size = len(tags)
learning_rate = 0.001
num_epochs = 200

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# print(Y_train)

# for epoch in range(num_epochs):
#     for (words, labels) in train_loader:
#         words = words.to(device)
#         labels = labels.to(device)

#         # forward pass
#         outputs = model(words)
#         loss = criterion(outputs, labels)

#         # backward and optimizer pass
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     if (epoch+1)%100 == 0:
#         print(f"epoch: {epoch+1}/{num_epochs}, loss:{loss.item():.4f}")

# print('------------------------------------------------------------')
# print(f"last epoch: {epoch+1}/{num_epochs}, loss:{loss.item():.4f}")
# print('------------------------------------------------------------')

# data = {
#     "model_state": model.state_dict(),
#     "input_size": input_size,
#     "output_size": output_size,
#     "hidden_size": hidden_size,
#     "all_words": all_words,
#     "tags": tags
# }

# FILE = "data.pth"
# torch.save(data, FILE)

# print(f"training complete, file saved in {FILE}")

loss_values = []

# Training loop
for epoch in range(num_epochs):
    total_loss = 0
    
    # Loop through batches
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)

        # Backward pass and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    # Calculate average loss for the epoch
    avg_loss = total_loss / len(train_loader)
    loss_values.append(avg_loss)
    
    # Print loss every 100 epochs
    if (epoch + 1) % 10 == 0:
        print(f"epoch: {epoch + 1}/{num_epochs}, loss:{avg_loss:.4f}")

print('------------------------------------------------------------')
print(f"Last epoch: {epoch + 1}/{num_epochs}, loss: {avg_loss:.4f}")
print('------------------------------------------------------------')

# Save model data
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f"Training complete, file saved in {FILE}")

# Plotting the loss function
plt.plot(loss_values)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Function During Training")
plt.show()