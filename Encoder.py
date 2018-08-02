import torch
import torch.nn as nn
from torch.autograd import Variable

class bilstm_encoder(nn.Module):
	def __init__(self, args):
		super(bilstm_encoder,self).__init__()
		self.args = args
		self.lstm = nn.LSTM(args.input_dim, args.bilstm_hidden_dim, num_layers=args.bilstm_n_layer, bidirectional=True)
	def forward(self, input_embeddings):
		hidden = self.inithidden()
		outputs, hidden = self.lstm(input_embeddings.unsqueeze(1), hidden)
		return outputs
	def inithidden(self):
		if self.args.gpu:
			result = (Variable(torch.zeros(2*self.args.bilstm_n_layer, 1, self.args.bilstm_hidden_dim)).cuda(),
				Variable(torch.zeros(2*self.args.bilstm_n_layer, 1, self.args.bilstm_hidden_dim)).cuda())
			return result
		else:
			result = (Variable(torch.zeros(2*self.args.bilstm_n_layer, 1, self.args.bilstm_hidden_dim)),
				Variable(torch.zeros(2*self.args.bilstm_n_layer, 1, self.args.bilstm_hidden_dim)))
			return result

class transformer_encoder(nn.Module):
	def __init__(self, args):
		super(transformer_encoder,self).__init__()
		self.args = args
		self.lstm = nn.LSTM(args.input_dim, args.bilstm_hidden_dim, num_layers=args.bilstm_n_layer, bidirectional=True)
	def forward(self, input_embeddings):
		hidden = self.inithidden()
		outputs, hidden = self.lstm(input_embeddings.unsqueeze(1), hidden)
		return outputs
	def inithidden(self):
		if self.args.gpu:
			result = (Variable(torch.zeros(2*self.args.bilstm_n_layer, 1, self.args.bilstm_hidden_dim)).cuda(),
				Variable(torch.zeros(2*self.args.bilstm_n_layer, 1, self.args.bilstm_hidden_dim)).cuda())
			return result
		else:
			result = (Variable(torch.zeros(2*self.args.bilstm_n_layer, 1, self.args.bilstm_hidden_dim)),
				Variable(torch.zeros(2*self.args.bilstm_n_layer, 1, self.args.bilstm_hidden_dim)))
			return result