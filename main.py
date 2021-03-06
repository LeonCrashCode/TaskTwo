import argparse
from dictionary import vocabulary
from dictionary import PretrainedEmb
from utils import read_input
from utils import get_singleton_dict
from utils import input2instance
from utils import read_output
from utils import output2action
from system import system_check_and_init

from representation import token_representation
from Encoder import bilstm_encoder
from Decoder import in_order_constituent_parser
from Decoder import in_order_constituent_parser_mask
from Decoder import in_order_constituent_action2tree
from eval import constituent_parser_eval

import torch

from optimizer import optimizer


def run_train(args, hypers):
	system_check_and_init(args)
	if args.gpu:
		print "GPU available"
	else:
		print "CPU only"

	word_v = vocabulary()
	char_v = vocabulary()
	actn_v = vocabulary()
	pretrain = PretrainedEmb(args.pretrain_path)

	#instances
	train_input = read_input(args.train_input)
	dev_input = read_input(args.dev_input)

	singleton_idx_dict, word_dict, word_v = get_singleton_dict(train_input, word_v)
	extra_vl = [ vocabulary() for i in range(len(train_input[0])-1)]	
	train_instance, word_v, char_v, extra_vl = input2instance(train_input, word_v, char_v, pretrain, extra_vl, word_dict, args, "train")
	word_v.freeze()
	char_v.freeze()
	for i in range(len(extra_vl)):
		extra_vl[i].freeze()
	dev_instance, word_v, char_v, extra_vl = input2instance(dev_input, word_v, char_v, pretrain, extra_vl, {}, args, "dev")

	train_output = read_output(args.train_action)
	dev_output = read_output(args.dev_action)
	train_action, actn_v = output2action(train_output, actn_v)
	#dev_actoin, actn_v = output2action(dev_output, actn_v)

	print "word vocabulary size:", word_v.size()
	print "char vocabulary size:", char_v.size() - 1
	print "pretrain vocabulary size:", pretrain.size() - 1
	extra_vl_size = []
	for i in range(len(extra_vl)):
		print "extra", i, "vocabulary size:", extra_vl[i].size()
		extra_vl_size.append(extra_vl[i].size())
	print "action vocaluary size:", actn_v.size() - 1
	actn_v.freeze()
	actn_v.dump()

	# neural components
	input_representation = token_representation(word_v.size(), char_v.size(), pretrain, extra_vl_size, args)
	encoder = None
	if args.encoder == "BILSTM":
		encoder = bilstm_encoder(args)
	elif args.encoder == "Transformer":
		encoder = transformer(args)
	assert encoder, "please specify encoder type"
	
	decoder = in_order_constituent_parser(actn_v.size(), actn_v.toidx("TERM"), args)
	mask = in_order_constituent_parser_mask(actn_v)

	if args.gpu:
		encoder = encoder.cuda()
		decoder = decoder.cuda()
		input_representation = input_representation.cuda()
	
	#training process
	model_parameters = list(encoder.parameters()) + list(decoder.parameters()) + list(input_representation.parameters())
	#model_optimizer = optimizer(args, model_parameters)
	lr = args.learning_rate_f

	i = len(train_instance)
	check_iter = 0
	check_loss = 0
	bscore = -1
	epoch = -1
	while True:
		for p in model_parameters:
			if p.grad is not None:
				p.grad.detach_()
				p.grad.zero_()
		
		if i == len(train_instance):
			i = 0
			epoch += 1
			lr = args.learning_rate_f / (1 + epoch * args.learning_rate_decay_f)

		check_iter += 1
		input_t = input_representation(train_instance[i], singleton_idx_dict=singleton_idx_dict, test=False)
		enc_rep_t = encoder(input_t, test=False)
		loss_t = decoder(enc_rep_t, mask, train_action[i], test=False)
		check_loss += loss_t.data.tolist()

		if check_iter % args.check_per_update == 0:
			print('epoch %.6f : %.10f [lr: %.6f]' % (check_iter*1.0/len(train_instance), check_loss*1.0 / args.check_per_update, lr))
			check_loss = 0
		
		if check_iter % args.eval_per_update == 0:
			trees = []
			for j, instance in enumerate(dev_instance):
				dev_input_embeddings = input_representation(instance)
				dev_enc_rep = encoder(dev_input_embeddings)
				dev_action_output = decoder(dev_enc_rep, mask)
				#print dev_action_output
				#print dev_input[j][0][1:-1]
				#print dev_input[j][-1][1:-1]
				trees.append(in_order_constituent_action2tree(dev_action_output, actn_v, dev_input[j][0][1:-1], dev_input[j][-1][1:-1]))
			with open("tmp/dev.output.tmp", "w") as w:
				for tree in trees:
					w.write(tree+"\n")
				w.flush()
				w.close()
			score = constituent_parser_eval(args)
			print('dev F-score %.10f ' % (score))
			if score >= bscore:
				bscore = score
				torch.save({"encoder":encoder.state_dict(), "decoder":decoder.state_dict(), "input_representation": input_representation.state_dict()}, args.model_path_base+"/model")
		i += 1
		loss_t.backward()
		torch.nn.utils.clip_grad_value_(model_parameters, 5)
		#model_optimizer.step()
		for p in model_parameters:
			if p.requires_grad:
				p.data.add_(-lr, p.grad.data)


def assign_hypers(subparser, hypers):
	for key in hypers.keys():
		if key[-3:] == "dim" or key[-5:] == "layer":
			subparser.add_argument("--"+key, default=int(hypers[key]))
		elif key[-4:] == "prob" or key[-2:] == "-f":
			subparser.add_argument("--"+key, default=float(hypers[key]))
		else:
			subparser.add_argument("--"+key, default=str(hypers[key]))

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	subparsers = parser.add_subparsers()

	hypers = {}
	for line in open("config"):
		line = line.strip()
		if line == "" or line[0] == "#":
			continue
		hypers[line.split()[0]] = line.split()[1]

	subparser = subparsers.add_parser("train")
	subparser.set_defaults(callback=lambda args: run_train(args, hypers))
	assign_hypers(subparser, hypers)
	subparser.add_argument("--numpy-seed", type=int)
	subparser.add_argument("--model-path-base", required=True)
	subparser.add_argument("--train-input", default="data/02-21.input")
	subparser.add_argument("--train-action", default="data/02-21.action")
	subparser.add_argument("--dev-input", default="data/22.input")
	subparser.add_argument("--dev-action", default="data/22.gold")
	subparser.add_argument("--dev-output", default="data/22.auto.clean.notop")
	subparser.add_argument("--batch-size", type=int, default=250)
	subparser.add_argument("--check-per-update", type=int, default=1000)
	subparser.add_argument("--eval-per-update", type=int, default=30000)
	subparser.add_argument("--eval-path-base", default="EVALB")
	subparser.add_argument("--encoder", default="BILSTM", help="BILSTM, Transformer")
	subparser.add_argument("--use-char", action='store_true')
	subparser.add_argument("--pretrain-path")
	subparser.add_argument("--gpu", action='store_true')
	subparser.add_argument("--optimizer", default="adam")

	args = parser.parse_args()
	args.callback(args)
