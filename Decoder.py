import torch
import torch.nn as nn
from torch.autograd import Variable

class in_order_constituent_parser(nn.Module):
	def __init__(self, action_size, args):
		super(in_order_constituent_parser,self).__init__()
		self.args = args
		self.lstm = nn.LSTM(args.action_dim, args.action_hidden_dim, num_layers=args.action_n_layer, bidirectional=False)
		self.action_embeds = nn.Embedding(action_size, args.action_dim)
	def forward(self, actions, masks, stack_masks, buffer_masks, enccoder_output, test=False):
		if not test:
			start_embeddings = self.initaction()
			action_variable = Variable(torch.LongTensor(actions[:-1]))
			if self.args.gpu:
				action_variable = action_variable.cuda()
			action_embeddings = self.action_embeds(action_variable)
			action_embeddings = torch.cat((start_embeddings, action_embeddings),0)
			hidden = self.inithidden()
			output, hidden = self.lstm(action_embeddings.unsqueeze(1), hidden)
			
			attn_scores = torch.bmm(output.transpose(0,1), encoder_output.transpose(0,1).transpose(1,2)).view(output.size(0),-1)
			attn_stack_weights = F.softmax(attn_scores + (stack_masks - 1) * 1e10, 1)
			attn_buffer_weights = F.softmax(attn_scores + (buffer_masks - 1) * 1e10, 1)
			attn_stack_hiddens = torch.bmm(attn_stack_weights.unsqueeze(0),encoder_output.transpose(0,1)).view(output.size(0),-1)
			attn_buffer_hiddens = torch.bmm(attn_buffer_weights.unsqueeze(0),encoder_output.transpose(0,1)).view(output.size(0),-1)
			feat_hiddens = self.feat_tanh(self.feat(torch.cat((attn_stack_hiddens, attn_buffer_hiddens, action_embeddings), 1)))
			dist = self.out(feat_hiddens)
			log_softmax_output = F.log_softmax(dist + (masks - 1) * 1e10, 1)
			return log_softmax_output

	def initaction(self):
		if self.args.gpu:
			return Variable(torch.zeros(1, self.args.action_dim)).cuda()
		else:
			return Variable(torch.zeros(1, self.args.action_dim))
	def initcharhidden(self):
		if self.args.gpu:
			result = (Variable(torch.zeros(self.args.action_n_layer, 1, self.args.action_hidden_dim)).cuda(),
				Variable(torch.zeros(self.args.action_n_layer, 1, self.args.action_hidden_dim)).cuda())
			return result
		else:
			result = (Variable(torch.zeros(self.args.action_n_layer, 1, self.args.action_hidden_dim)),
				Variable(torch.zeros(self.args.action_n_layer, 1, self.args.action_hidden_dim)))
			return result
class in_order_constituent_parser_mask:
	def __init__(self, actn_v):
		self.actn_v = actn_v
		pass
	def init(self, sentence_size):
		self.stack = []
		self.buffer = sentence_size - 2 # ignore dummy tokens (start and end)
		self.size = sentence_size - 2 
		self.unary = 0
		self.prev_a = ""
		self.open_non_terminal_node = 0
	def get_mask(self, actions):
		masks = []
		stack_masks = []
		buffer_masks = []
		for a in actions:
			masks.append(self.get_step_mask())
			stack_masks.append(self.get_stack_mask())
			buffer_masks.append(self.get_buffer_mask())
			assert masks[-1][a] == 1, "mask error"
			self.update(a)
	def update(self, a):
		# 0 is terminal node
		# 1 is open non-terminal node
		# 2 is close non-terminal node
		a = self.actn_v.totok(a)
		if a == "SHIFT":
			self.stack.append(0)
			self.buffer -= 1
			self.unary = 0
		elif a[:2] == "PJ":
			self.stack[-1] = 1
			self.unary += 1
			self.open_non_terminal_node += 1
		elif a == "REDUCE":
			self.open_non_terminal_node -= 1
			while True:
				if self.stack[-1] == 0 or self.stack[-1] == 2:
					self.stack.pop()
				elif self.stack[-1] == 1:
					self.stack[-1] = 2
					break
				else:
					assert False, "no allowed state"
			if self.prev_a == "REDUCE":
				self.unary = 0
		elif a == "TERM":
			pass
		else:
			assert False, "no allowed action"
		self.prev_a = a
	def get_step_mask(self):
		mask = []
		for i in range(self.actn_v.size()):
			a = self.actn_v.totok(i)
			if i == 0:
				mask.append(0)
			elif a == "SHIFT":
				if self.buffer == 0:
					mask.append(0)
				elif len(self.stack) == 0:
					mask.append(1)
				elif self.open_non_terminal_node == 0:
					mask.append(0)
				else:
					mask.append(1)
			elif a[:2] == "PJ":
				if len(self.stack) == 0:
					mask.append(0)
				elif self.stack[-1] == 1:
					mask.append(0)
				elif self.unary >= 5:
					mask.append(0)
				else:
					mask.append(1)
			elif a == "REDUCE":
				if self.open_non_terminal_node == 0:
					mask.append(0)
				else:
					mask.append(1)
			elif a == "TERM":
				if self.open_non_terminal_node == 0 and len(self.stack) == 1 and self.buffer == 0:
					mask.append(1)
				else:
					mask.append(0)
			else:
				assert False, "no allowed action"
		return mask
	def get_stack_mask():
		return [1] + [ 1 for i in range(self.size - self.buffer)] + [ 0 for i in range(self.buffer)] + [0]
	def get_buffer_mask():
		return [0] + [ 0 for i in range(self.size - self.buffer)] + [ 1 for i in range(self.buffer)] + [1]

				




