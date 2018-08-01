import torch
import torch.nn as nn
from torch.autograd import Variable
class token_representation(nn.Module):
	def __init__(self, word_size, char_size, pretrain, extra_vl_size, args):
		super(token_representation,self).__init__()
		self.args = args
		self.word_embeds = nn.Embedding(word_size, args.word_dim)
		if args.use_char:
			self.char_embeds = nn.Embedding(char_size, args.char_dim)
			self.lstm = nn.LSTM(args.char_dim, args.char_hidden_dim, num_layers=args.char_n_layers, bidirectional=True)
		if args.pretrain_path:
			self.pretrain_embeds = nn.Embedding(pretrain.size(), args.pretrain_dim)
			self.pretrain_embeds.weight = nn.Parameter(pretrain.vectors, False)
		if args.extra_dim:
			dims = args.extra_dim.split()
			self.extra_embeds = []
			for i, size in enumerate(extra_vl_size):
				self.extra_embeds.append(nn.Embedding(size, int(dims[i])))

	def forward(self, instance, singleton_idx_dict, test=False, gpu=True):
		if not test:
			word_sequence = []
			for i, widx in enumerate(instance[0], 0):
				if (widx in singleton_idx_dict) and random.random() < 0.1:
					word_sequence.append(instance[3][i])
				else:
					word_sequence.append(widx)
		word_variable = Variable(torch.LongTensor(word_sequence), volatile=test)
		if gpu:
			word_variable = word_variable.cuda()
		word_embeddings = self.word_embeds(word_variable)
		print word_embeddings
		exit(1)
		if self.args.use_char:
			char_outputs = []
			for char_instance in instance[1]:
				char_variable = Variable(torch.LongTensor(char_instance), volatile=test)
				if gpu:
					char_variable = char_variable.cuda()
				char_embeddings = self.char_embeds(char_variable)
				char_hidden = self.initcharhidden(gpu)
				char_output, char_hidden = self.lstm(char_embeddings, char_hidden)
				char_outputs.append(char_output)
			char_embeddings = torch.cat(char_outputs)
			word_embeddings = torch.cat((word_embeddings, char_embeddings))

		if self.args.pretrain_path:
			pretrain_variable = Variable(torch.LongTensor(instance[2]), volatile=test)
			if gpu:
				pretrain_variable = pretrain_variable.cuda()
			pretrain_embeddings = self.pretrain_embeds(pretrain_variable)
			word_embeddings = torch.cat((word_embeddings, pretrain_embeddings))

		return word_embeddings

	def initcharhidden(self, gpu):
		if gpu:
			result = (Variable(torch.zeros(2*self.args.char_n_layers, 1, self.args.char_hidden_dim)).cuda(),
				Variable(torch.zeros(2*self.args.char_n_layers, 1, self.args.char_hidden_dim)).cuda())
			return result
		else:
			result = (Variable(torch.zeros(2*self.args.char_n_layers, 1, self.args.char_hidden_dim)),
				Variable(torch.zeros(2*self.args.char_n_layers, 1, self.args.char_hidden_dim)))
			return result