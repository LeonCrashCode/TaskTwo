import torch
import torch.nn as nn
import torch.nn.functional as F

class in_order_constituent_parser(nn.Module):
	def __init__(self, action_size, terminal_action_idx, args):
		super(in_order_constituent_parser,self).__init__()
		self.args = args
		self.lstm = nn.LSTM(args.action_dim, args.action_hidden_dim, num_layers=args.action_n_layer, bidirectional=False)
		self.action_embeds = nn.Embedding(action_size, args.action_dim)
		self.feat = nn.Linear(args.action_hidden_dim * 2 + args.action_dim, args.action_feature_dim)
		self.feat_tanh = nn.Tanh()
		self.out = nn.Linear(args.action_feature_dim, action_size)
		self.dropout = nn.Dropout(args.dropout_f)
		self.criterion = nn.NLLLoss()
		self.terminal_action_idx = terminal_action_idx
	def forward(self, encoder_output_t, mask, actions=None, test=True):
		mask.init(encoder_output_t.size(0))
		if not test:
			self.lstm.dropout = self.args.dropout_f
			masks, stack_masks, buffer_masks = mask.get_mask(actions)
			masks_t = torch.FloatTensor(masks)
			stack_masks_t = torch.FloatTensor(stack_masks)
			buffer_masks_t = torch.FloatTensor(buffer_masks)

			if self.args.gpu:
				masks_t = masks_t.cuda()
				stack_masks_t = stack_masks_t.cuda()
				buffer_masks_t = buffer_masks_t.cuda()

			start_t = self.initaction()
			action_t = torch.LongTensor(actions[:-1])
			if self.args.gpu:
				action_t = action_t.cuda()
			action_t = self.action_embeds(action_t)
			action_t = torch.cat((start_t, action_t),0)
			action_t = self.dropout(action_t)
			hidden_t = self.inithidden()
			output_t, _ = self.lstm(action_t.unsqueeze(1), hidden_t)

			attn_scores_t = torch.bmm(output_t.transpose(0,1), encoder_output_t.transpose(0,1).transpose(1,2)).view(output_t.size(0),-1)
			attn_stack_weights_t = F.log_softmax(attn_scores_t + (stack_masks_t - 1) * 1e10, 1)
			attn_buffer_weights_t = F.log_softmax(attn_scores_t + (buffer_masks_t - 1) * 1e10, 1)
			attn_stack_hiddens_t = torch.bmm(attn_stack_weights_t.unsqueeze(0),encoder_output_t.transpose(0,1)).view(output_t.size(0),-1)
			attn_buffer_hiddens_t = torch.bmm(attn_buffer_weights_t.unsqueeze(0),encoder_output_t.transpose(0,1)).view(output_t.size(0),-1)
			feat_hiddens_t = self.feat_tanh(self.feat(torch.cat((attn_stack_hiddens_t, attn_buffer_hiddens_t, action_t), 1)))
			dist_t = self.out(feat_hiddens_t)
			log_softmax_output_t = F.log_softmax(dist_t + (masks_t - 1) * 1e10, 1)

			action_g_t = torch.LongTensor(actions)
			if self.args.gpu:
				action_g_t = action_g_t.cuda()
			#print log_softmax_output_t, log_softmax_output_t.size()

			loss_t = self.criterion(log_softmax_output_t, action_g_t)
			return loss_t, None
		else:
			self.lstm.dropout = 0
			action_t = self.initaction()
			hidden_t = self.inithidden()

			action_output = []
			while True:
				masks = mask.get_step_mask()
				stack_masks = mask.get_stack_mask()
				buffer_masks = mask.get_buffer_mask()

				masks_t = torch.FloatTensor(masks)
				stack_masks_t = torch.FloatTensor(stack_masks)
				buffer_masks_t = torch.FloatTensor(buffer_masks)
				#print masks
				#print stack_masks
				#print buffer_masks
				if self.args.gpu:
					masks_t = masks_t.cuda()
					stack_masks_t = stack_masks_t.cuda()
					buffer_masks_t = buffer_masks_t.cuda()

				output_t, hidden_t = self.lstm(action_t.unsqueeze(1), hidden_t)
				attn_scores_t = torch.bmm(output_t.transpose(0,1), encoder_output_t.transpose(0,1).transpose(1,2)).view(output_t.size(0),-1)
				attn_stack_weights_t = F.log_softmax(attn_scores_t + (stack_masks_t - 1) * 1e10, 1)
				attn_buffer_weights_t = F.log_softmax(attn_scores_t + (buffer_masks_t - 1) * 1e10, 1)
				attn_stack_hiddens_t = torch.bmm(attn_stack_weights_t.unsqueeze(0),encoder_output_t.transpose(0,1)).view(output_t.size(0),-1)
				attn_buffer_hiddens_t = torch.bmm(attn_buffer_weights_t.unsqueeze(0),encoder_output_t.transpose(0,1)).view(output_t.size(0),-1)
				feat_hiddens_t = self.feat_tanh(self.feat(torch.cat((attn_stack_hiddens_t, attn_buffer_hiddens_t, action_t), 1)))
				dist_t = self.out(feat_hiddens_t).view(-1)
				dist_t = dist_t + (masks_t - 1) * 1e10
				dist_l = dist_t.tolist()
				action_idx = max(enumerate(dist_l), key=lambda x: x[1])[0]
				action_output.append(action_idx)
				if action_idx == self.terminal_action_idx:
					break
				action_t = torch.LongTensor([action_idx])
				if self.args.gpu:
					action_t = action_t.cuda()
				action_t = self.action_embeds(action_t)
				mask.update(action_idx)
				#print action_idx
			return action_output


	def initaction(self):
		if self.args.gpu:
			return torch.zeros(1, self.args.action_dim, requires_grad=True).cuda()
		else:
			return torch.zeros(1, self.args.action_dim, requires_grad=True)
	def inithidden(self):
		if self.args.gpu:
			result = (torch.zeros(self.args.action_n_layer, 1, self.args.action_hidden_dim, requires_grad=True).cuda(),
				torch.zeros(self.args.action_n_layer, 1, self.args.action_hidden_dim, requires_grad=True).cuda())
			return result
		else:
			result = (torch.zeros(self.args.action_n_layer, 1, self.args.action_hidden_dim, requires_grad=True),
				torch.zeros(self.args.action_n_layer, 1, self.args.action_hidden_dim, requires_grad=True))
			return result
class in_order_constituent_parser_mask:
	def __init__(self, actn_v):
		self.actn_v = actn_v
		pass
	def init(self, sentence_size):
		self.stack = [0]	
		self.buffer = sentence_size - 2 + 1# ignore dummy tokens (start and end)
		self.size = sentence_size - 2 + 1
		self.unary = 0
		self.prev_a = ""
		self.nopen_parens = 0
		self.is_open_paren = []
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
		return masks, stack_masks, buffer_masks
	def update(self, a):
		# 0 is terminal node
		# 1 is open non-terminal node
		# 2 is close non-terminal node
		a = self.actn_v.totok(a)
		if a == "SHIFT":
			self.stack.append(0)
			self.buffer -= 1
			self.unary = 0
			self.is_open_paren.append(-1)
		elif a[:2] == "PJ":
			self.stack.append(-1)
			self.nopen_parens += 1
			self.is_open_paren.append(0)
		elif a == "REDUCE":
			self.nopen_parens -= 1
			if self.prev_a == "REDUCE":
				self.unary = 0
			if self.prev_a[:2] == "PJ":
				self.unary += 1

			i = len(self.is_open_paren) - 1
			while self.is_open_paren[i] < 0:
				i -= 1

			nchildren = len(self.is_open_paren) - i - 1
			for i in range(nchildren):
				self.stack.pop()
				self.is_open_paren.pop()
			self.is_open_paren = self.is_open_paren[:-2]
			self.stack = self.stack[:-2]
			self.stack.append(0)
			self.is_open_paren.append(-1)
		elif a == "TERM":
			pass
		else:
			print a
			assert False, "no allowed action"
		self.prev_a = a
	def illegal(self, a, prev_a, ssize, bsize, nopen_parens, unary):
		if a == "<UNK>":
			return True
		if a == "TERM":
			if ssize == 2 and bsize == 1 and prev_a == "REDUCE":
				return False
			return True

		if ssize == 1:
			if a != "SHIFT":
				return True
			else:
				return False
		if a == "SHIFT":
			if bsize == 1:
				return True
			if nopen_parens == 0:
				return True
			return False
		if a[:2] == "PJ":
			if bsize == 1 and unary >= 3:
				return True
			if prev_a[:2] == "PJ":
				return True
			return False
		if a == "REDUCE":
			if unary > 3:
				return True
			if nopen_parens == 0:
				return True
			return False
	def get_step_mask(self):
		mask = []
		for i in range(self.actn_v.size()):
			a = self.actn_v.totok(i)
			if self.illegal(a, self.prev_a, len(self.stack), self.buffer, self.nopen_parens, self.unary):
				mask.append(0)
			else:
				mask.append(1)
		return mask
	def get_stack_mask(self):
		return [1] + [ 1 for i in range(self.size - self.buffer)] + [ 0 for i in range(self.buffer - 1)] + [0]
	def get_buffer_mask(self):
		return [0] + [ 0 for i in range(self.size - self.buffer)] + [ 1 for i in range(self.buffer - 1)] + [1]

				




