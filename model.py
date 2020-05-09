import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, batch_first = True)
        self.fc = nn.Linear(hidden_size, vocab_size) 
        #self.drop = nn.Dropout(p=0.2)
        self.hidden = self.init_hidden()

        
    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_size),
                torch.zeros(1, 1, self.hidden_size))

    
    def forward(self, features, captions):
        captions = captions[:,:-1]
        captions = self.word_embeddings(captions)
        inputs = torch.cat((features.unsqueeze(1), captions),1)

        out, self.hidden = self.lstm(inputs,self.hidden)
        print(out.shape)
        #out = self.drop(out)
        out = self.fc(out)

        return out

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        sentence = []
        for i in range(max_len):
            lstm_out, states = self.lstm(inputs, states)       
            out = self.fc(lstm_out.squeeze(1))
            aa, predicted = out.max(dim=1)
            sentence.append(predicted.item())
            inputs = self.word_embeddings(predicted)  
            inputs = inputs.unsqueeze(1)
        return sentence