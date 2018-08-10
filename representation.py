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
		self.dropout = nn.Dropout(self.args.dropout_f)

	def forward(self, instance, singleton_idx_dict=None, test=True):
		if not test:
			word_sequence = []
			for i, widx in enumerate(instance[0], 0):
				if (widx in singleton_idx_dict) and random() < 0.1:
					word_sequence.append(instance[3][i])
				else:
					word_sequence.append(widx)
		word_t = torch.LongTensor(word_sequence)
		if self.args.gpu:
			word_t = word_t.cuda()
		word_t = self.word_embeds(word_t)
		if not test:
			word_t = self.dropout(word_t)

		if self.args.use_char:
			char_ts = []
			for char_instance in instance[1]:
				char_t = torch.LongTensor(char_instance)
				if self.args.gpu:
					char_t = char_t.cuda()
				char_t = self.char_embeds(char_t)
				char_hidden_t = self.initcharhidden()
				char_t, _ = self.lstm(char_t.unsqueeze(1), char_hidden_t)
				char_t_per_word = torch.sum(char_t,0)
				if not test:
					char_t_per_word = self.dropout(char_t_per_word)
				char_ts.append(char_t_per_word)
			char_t = torch.cat(char_ts, 0)
			word_t = torch.cat((word_t, char_t), 1)
		#print word_embeddings, word_embeddings.size()
		if self.args.pretrain_path:
			pretrain_t = torch.LongTensor(instance[2])
			if self.args.gpu:
				pretrain_t = pretrain_t.cuda()
			pretrain_t = self.pretrain_embeds(pretrain_t)
			word_t = torch.cat((word_t, pretrain_t), 1)
		#print word_embeddings, word_embeddings.size()
		if self.args.extra_dim_list:
			for i, extra_embeds in enumerate(self.extra_embeds):
				extra_t = torch.LongTensor(instance[4+i])
				if self.args.gpu:
					extra_t = extra_t.cuda()
				extra_t = extra_embeds(extra_t)
				if not test:
					extra_t = self.dropout(extra_t)
				word_t = torch.cat((word_t, extra_t), 1)
		#print word_embeddings, word_embeddings.size()
		word_t = self.tanh(self.info2input(word_t))
		#print word_embeddings, word_embeddings.size()
		return word_t

	def initcharhidden(self):
		if self.args.gpu:
			result = (torch.zeros(2*self.args.char_n_layer, 1, self.args.char_hidden_dim, requires_grad=True).cuda(),
				torch.zeros(2*self.args.char_n_layer, 1, self.args.char_hidden_dim, requires_grad=True).cuda())
			return result
		else:
			result = (torch.zeros(2*self.args.char_n_layer, 1, self.args.char_hidden_dim, requires_grad=True),
				torch.zeros(2*self.args.char_n_layer, 1, self.args.char_hidden_dim, requires_grad=True))
			return result