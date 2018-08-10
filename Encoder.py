import torch
import torch.nn as nn
from torch.autograd import Variable

class bilstm_encoder(nn.Module):
	def __init__(self, args):
		super(bilstm_encoder,self).__init__()
		self.args = args
		self.lstm = nn.LSTM(args.input_dim, args.bilstm_hidden_dim, num_layers=args.bilstm_n_layer, bidirectional=True)
	def forward(self, input_t, test=False):
		hidden_t = self.inithidden()
		if not test:
			self.lstm.dropout = self.args.dropout_f
		else:
			self.lstm.dropout = 0
		output_t, _ = self.lstm(input_t.unsqueeze(1), hidden_t)
		return output_t
	def inithidden(self):
		if self.args.gpu:
			result = (torch.zeros(2*self.args.bilstm_n_layer, 1, self.args.bilstm_hidden_dim, requires_grad=True).cuda(),
				torch.zeros(2*self.args.bilstm_n_layer, 1, self.args.bilstm_hidden_dim, requires_grad=True).cuda())
			return result
		else:
			result = (torch.zeros(2*self.args.bilstm_n_layer, 1, self.args.bilstm_hidden_dim, requires_grad=True),
				torch.zeros(2*self.args.bilstm_n_layer, 1, self.args.bilstm_hidden_dim, requires_grad=True))
			return result

