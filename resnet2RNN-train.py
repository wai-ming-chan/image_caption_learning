#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms

# !pip install transformers
from transformers import BertTokenizer

# !pip install utils
# from utils import save_checkpoint, load_checkpoint, print_examples
from torch import optim, nn
from tqdm import tqdm
import torch
import torch.nn as nn
import statistics
from torchvision.models import resnet50, ResNet50_Weights

import time

import torchvision.transforms as transforms
from PIL import Image



# BLEU score
from torcheval.metrics import BLEUScore

def getBLEUscores(model, device, dataset):
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
   
    model.eval()
   
    test_img1 = transform(Image.open("data/test_examples/dog.jpg").convert("RGB")).unsqueeze(0)
    references = ["<SOS> dog on a beach by the ocean . <EOS>"]
    candidates = "" + " ".join( model.caption_image(test_img1.to(device), dataset.vocab) )
#    candidates = "<SOS> Dog on a beach by the ocean . <EOS>"

#    candidates = candidates.removesuffix(" . <EOS>").removeprefix("<SOS> ")
#    metric.update( candidates, reference)
#    score1 = metric.compute().item()
#    print("score1: ", score1)

    scores_img1 = []
    for bleu_idx in range(1, 5):
        metric = BLEUScore(n_gram=bleu_idx)
        metric.update( candidates, references )
        scores_img1.append( metric.compute().item() )
    print("Example 1: scores=", scores_img1)


    test_img2 = transform(Image.open("data/test_examples/child.jpg").convert("RGB")).unsqueeze(0)
    references = ["<SOS> child holding red frisbee outdoors . <EOS>"]
    candidates = "" + " ".join( model.caption_image(test_img2.to(device), dataset.vocab) )

    scores_img2 = []
    for bleu_idx in range(1, 5):
        metric = BLEUScore(n_gram=bleu_idx)
        metric.update( candidates, references )
        scores_img2.append( metric.compute().item() )
    print("Example 2: scores=", scores_img2)



    test_img3 = transform(Image.open("data/test_examples/bus.png").convert("RGB")).unsqueeze(0)
    references = ["<SOS> bus driving by parked cars . <EOS>"]
    candidates = "" + " ".join( model.caption_image(test_img3.to(device), dataset.vocab) )

    scores_img3 = []
    for bleu_idx in range(1, 5):
        metric = BLEUScore(n_gram=bleu_idx)
        metric.update( candidates, references )
        scores_img3.append( metric.compute().item() )
    print("Example 3: scores=", scores_img3)


    
    test_img4 = transform(Image.open("data/test_examples/boat.png").convert("RGB")).unsqueeze(0)
    references = ["<SOS> a small boat in the ocean . <EOS>"]
    candidates = "" + " ".join( model.caption_image(test_img4.to(device), dataset.vocab) )

    scores_img4 = []
    for bleu_idx in range(1, 5):
        metric = BLEUScore(n_gram=bleu_idx)
        metric.update( candidates, references )
        scores_img4.append( metric.compute().item() )
    print("Example 4: scores=", scores_img4)



    test_img5 = transform(Image.open("data/test_examples/horse.png").convert("RGB")).unsqueeze(0)
    references = ["<SOS> a cowboy riding a horse in the desert . <EOS>"]
    candidates = "" + " ".join( model.caption_image(test_img5.to(device), dataset.vocab) )

    scores_img5 = []
    for bleu_idx in range(1, 5):
        metric = BLEUScore(n_gram=bleu_idx)
        metric.update( candidates, references )
        scores_img5.append( metric.compute().item() )
    print("Example 5: scores=", scores_img5)

    model.train()


def print_examples(model, device, dataset):
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    model.eval()
    
    test_img1 = transform(Image.open("data/test_examples/dog.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 1 CORRECT: Dog on a beach by the ocean")
    print(
        "Example 1 OUTPUT: "
        + " ".join(model.caption_image(test_img1.to(device), dataset.vocab))
    )
    
    test_img2 = transform(
        Image.open("data/test_examples/child.jpg").convert("RGB")
    ).unsqueeze(0)
    print("Example 2 CORRECT: Child holding red frisbee outdoors")
    print(
        "Example 2 OUTPUT: "
        + " ".join(model.caption_image(test_img2.to(device), dataset.vocab))
    )
    
    test_img3 = transform(Image.open("data/test_examples/bus.png").convert("RGB")).unsqueeze(
        0
    )
    print("Example 3 CORRECT: Bus driving by parked cars")
    print(
        "Example 3 OUTPUT: "
        + " ".join(model.caption_image(test_img3.to(device), dataset.vocab))
    )
    
    test_img4 = transform(
        Image.open("data/test_examples/boat.png").convert("RGB")
    ).unsqueeze(0)
    print("Example 4 CORRECT: A small boat in the ocean")
    print(
        "Example 4 OUTPUT: "
        + " ".join(model.caption_image(test_img4.to(device), dataset.vocab))
    )
    
    test_img5 = transform(
        Image.open("data/test_examples/horse.png").convert("RGB")
    ).unsqueeze(0)
    print("Example 5 CORRECT: A cowboy riding a horse in the desert")
    print(
        "Example 5 OUTPUT: "
        + " ".join(model.caption_image(test_img5.to(device), dataset.vocab))
    )
    
    model.train()


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
#    torch.save(state, filename)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename_TT = f"my_checkpoint_{timestamp}.pth.tar"
    torch.save(state, filename_TT)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step




tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}  # integer to string
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}  # STRING TO INTEGER
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4

        for sentence in sentence_list:
            # tokenized_sentence = [tok.text.lower() for tok in spacy_eng.tokenizer(sentence)]
            tokenized_sentence = tokenizer.tokenize(sentence)
            for word in tokenized_sentence:
                frequencies[word] = frequencies.get(word, 0) + 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        # tokenized_text = [tok.text.lower() for tok in spacy_eng.tokenizer(text)]
        tokenized_text = tokenizer.tokenize(text)

        return [self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
                for token in tokenized_text]


class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.transform = transform

        self.imgs = self.df["image"]
        self.captions = self.df["caption"]

        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = self.imgs[index]
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        return img, torch.tensor(numericalized_caption)


transform = transforms.Compose([

    transforms.Resize((356, 356)),
    transforms.RandomCrop((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
print("os.getcwd(): ", os.getcwd())
dataset = FlickrDataset(root_dir='data/flickr8k/images',
                        captions_file='data/flickr8k/captions.txt',
                        transform=transform,
                        freq_threshold=5)



class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        images = [item[0].unsqueeze(0) for item in batch]
        images = torch.cat(images, dim=0)

        # Pad the sequences with zeros to make them the same length
        lengths = [len(sample[1]) for sample in batch]
        max_length = max(lengths)
        padded_batch = torch.full((len(batch), max_length), self.pad_idx, dtype=torch.long)
        for i, sample in enumerate(batch):
            padded_batch[i, :len(sample[1])] = torch.LongTensor(sample[1])

        targets = padded_batch

        return images, targets


pad_idx = dataset.vocab.stoi["<PAD>"]

train_loader = DataLoader(
    dataset=dataset,
    batch_size=64,
    shuffle=True,
    collate_fn=MyCollate(pad_idx=pad_idx))


# x, y = next(iter(train_loader))
# print(x.shape)
# print(y.shape)
# print()




class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(EncoderCNN, self).__init__()
        self.train_CNN = train_CNN
        # self.inception = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.times = []
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):

        for name, params in self.resnet.named_parameters():
            if "fc.weight" in name or "fc.bias" in name:
                params.required_grad = True
            else:
                params.required_grad = self.train_CNN

        features = self.resnet(images)
        return self.dropout(self.relu(features))


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs


# encoder = EncoderCNN(512)
# print(encoder(x).shape)
# decoder = DecoderRNN(512, hidden_size=256, vocab_size=len(dataset.vocab), num_layers=1)
# decoder(encoder(x), y).shape


class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CNNtoRNN, self).__init__()
        self.encoderCNN = EncoderCNN(embed_size)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features, captions)
        return outputs

    def caption_image(self, image, vocabulary, max_length=50):
        result_caption = []

        with torch.no_grad():
            x = self.encoderCNN(image).unsqueeze(1)
            states = None

            for _ in range(max_length):
                hiddens, states = self.decoderRNN.lstm(x, states)
                output = self.decoderRNN.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)
                result_caption.append(predicted.item())
                x = self.decoderRNN.embed(predicted).unsqueeze(0)

                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break

        return [vocabulary.itos[idx] for idx in result_caption]


# model = CNNtoRNN(embed_size=512, hidden_size=256, vocab_size=len(dataset.vocab), num_layers=1)
# model(x, y)
# print(model.caption_image(x[0].unsqueeze(0), dataset.vocab))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_model = True
save_model = True
train_CNN = True

# Hyperparameters
embed_size = 512
hidden_size = 512
vocab_size = len(dataset.vocab)
num_layers = 1
learning_rate = 3e-4
num_epochs = 200

# initialize model, loss etc
model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

step = 0
if load_model:
    step = load_checkpoint(torch.load("my_checkpoint_20230505-165753.pth.tar"), model, optimizer)

model.train()
loss_epoch=[]
for epoch in range(num_epochs):
    print(f'epoch {epoch + step} starts running...')
    loss_list=[]
    for idx, (imgs, captions) in tqdm(
            enumerate(train_loader), total=len(train_loader), leave=False
    ):
        imgs = imgs.to(device)
        captions = captions.to(device)

        outputs = model(imgs, captions[:, :-1])
        loss = criterion(
            outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1)
        )
        loss_list.append(loss.item())

        optimizer.zero_grad()
        loss.backward(loss)
        optimizer.step()
    loss_epoch.append(np.mean(loss_list))

    print()
    print(f'-------------->epoch {epoch + step}/{num_epochs} loss : {np.mean(loss_list)}<-------------------')
    getBLEUscores(model, device, dataset)

    # Uncomment the line below to see a couple of test cases
    print_examples(model, device, dataset)


    if save_model:
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": epoch+step
        }
        save_checkpoint(checkpoint)

getBLEUscores(model, device, dataset)
