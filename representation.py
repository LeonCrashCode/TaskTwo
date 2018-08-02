from random import random

import torch
import torch.nn as nn
from torch.autograd import Variable
class token_representation(nn.Module):
	def __init__(self, word_size, char_size, pretrain, extra_vl_size, args):
		super(token_representation,self).__init__()
		self.args = args
		self.word_embeds = nn.Embedding(word_size, args.word_dim)
		info_dim = args.word_dim
		if args.use_char:
			self.char_embeds = nn.Embedding(char_size, args.char_dim)
			self.lstm = nn.LSTM(args.char_dim, args.char_hidden_dim, num_layers=args.char_n_layer, bidirectional=True)
			info_dim += args.char_hidden_dim*2
		if args.pretrain_path:
			self.pretrain_embeds = nn.Embedding(pretrain.size(), args.pretrain_dim)
			self.pretrain_embeds.weight = nn.Parameter(torch.FloatTensor(pretrain.vectors()), False)
			info_dim += args.pretrain_dim
		if args.extra_dim_list:
			dims = args.extra_dim_list.split(",")
			self.extra_embeds = []
			for i, size in enumerate(extra_vl_size):
				self.extra_embeds.append(nn.Embedding(size, int(dims[i])))
				info_dim += int(dims[i])

		self.info2input = nn.Linear(info_dim, args.input_dim)
		self.tanh = nn.Tanh()

	def forward(self, instance, singleton_idx_dict, test=False):
		if not test:
			word_sequence = []
			for i, widx in enumerate(instance[0], 0):
				if (widx in singleton_idx_dict) and random() < 0.1:
					word_sequence.append(instance[3][i])
				else:
					word_sequence.append(widx)
		word_variable = Variable(torch.LongTensor(word_sequence), volatile=test)
		if self.args.gpu:
			word_variable = word_variable.cuda()
		word_embeddings = self.word_embeds(word_variable)
		#print word_embeddings, word_embeddings.size()
		if self.args.use_char:
			char_outputs = []
			for char_instance in instance[1]:
				char_variable = Variable(torch.LongTensor(char_instance), volatile=test)
				if self.args.gpu:
					char_variable = char_variable.cuda()
				char_embeddings = self.char_embeds(char_variable)
				char_hidden = self.initcharhidden()
				char_output, char_hidden = self.lstm(char_embeddings.unsqueeze(1), char_hidden)
				char_outputs.append(torch.sum(char_output,0))
			char_embeddings = torch.cat(char_outputs, 0)
			word_embeddings = torch.cat((word_embeddings, char_embeddings), 1)
		#print word_embeddings, word_embeddings.size()
		if self.args.pretrain_path:
			pretrain_variable = Variable(torch.LongTensor(instance[2]), volatile=test)
			if self.args.gpu:
				pretrain_variable = pretrain_variable.cuda()
			pretrain_embeddings = self.pretrain_embeds(pretrain_variable)
			word_embeddings = torch.cat((word_embeddings, pretrain_embeddings), 1)
		#print word_embeddings, word_embeddings.size()
		if self.args.extra_dim_list:
			for i, extra_embeds in enumerate(self.extra_embeds):
				extra_variable = Variable(torch.LongTensor(instance[4+i]), volatile=test)
				if self.args.gpu:
					extra_variable = extra_variable.cuda()
				extra_embeddings = extra_embeds(extra_variable)
				word_embeddings = torch.cat((word_embeddings, extra_embeddings), 1)
		#print word_embeddings, word_embeddings.size()
		word_embeddings = self.tanh(self.info2input(word_embeddings))
		#print word_embeddings, word_embeddings.size()
		return word_embeddings

	def initcharhidden(self):
		if self.args.gpu:
			result = (Variable(torch.zeros(2*self.args.char_n_layer, 1, self.args.char_hidden_dim)).cuda(),
				Variable(torch.zeros(2*self.args.char_n_layer, 1, self.args.char_hidden_dim)).cuda())
			return result
		else:
			result = (Variable(torch.zeros(2*self.args.char_n_layer, 1, self.args.char_hidden_dim)),
				Variable(torch.zeros(2*self.args.char_n_layer, 1, self.args.char_hidden_dim)))
			return result