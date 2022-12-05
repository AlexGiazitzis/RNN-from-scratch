#!/usr/bin/python
# -*- coding: utf-8 -*-

from RNN import RNN, accuracy, nll_loss
from utils import one_hot, parse_text, vec2word, word2idx

# RNN training routine for a simple alphabet sequence.
text = "abcdefghijklmnopqrstuvwxyz"

vocab, data = parse_text(text)
one_hot_data = one_hot(data, vocab)
idx_data = word2idx(data, vocab)

input_dim = len(vocab)
hidden_state_size = 32
epochs = 65
learning_rate = 1e-4

best_acc = 0;
best_loss = 1e15;
rnn = RNN(input_dim, hidden_state_size)
for epoch in range(epochs):
    print(f"Epoch {epoch}:")
    out_preds, _, out_probs = rnn(one_hot_data)
    
    total_loss, loss_t = nll_loss(out_probs, idx_data[0])
    print(f"\tloss := {total_loss:.2f}", end="")

    acc = accuracy(out_preds, idx_data[0])
    print(f"\t accuracy := {acc:.2f}%")
    if acc >= best_acc and total_loss <= best_loss:
        print("Saving model...")
        rnn.save("./model.npz")
        best_acc = acc
        best_loss = total_loss
        print("Model saved!")
    
    rnn.bppt(out_probs, idx_data[0].tolist(), one_hot_data, 5e-1, 1)

# Inference
rnn = RNN.load("./model.npz")
rnn.eval()

x_val = one_hot(['a'], vocab)
pred, *_ = rnn(x_val, length=30)
pred_text = vec2word(pred, vocab)

print(f"Validation text:\n{pred_text}")
