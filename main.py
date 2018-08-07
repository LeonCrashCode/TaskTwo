import argparse
from dictionary import vocabulary
from dictionary import PretrainedEmb
from utils import read_input
from utils import get_singleton_dict
from utils import input2instance
from utils import read_output
from utils import output2action

from representation import token_representation
from Encoder import bilstm_encoder
from Decoder import in_order_constituent_parser
from Decoder import in_order_constituent_parser_mask

import torch
use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.cuda.manual_seed_all(12345678)
torch.manual_seed(12345678)

def run_train(args, hypers):
	
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
	dev_instance, word_v, char_v, extra_vl = input2instance(train_input, word_v, char_v, pretrain, extra_vl, {}, args, "dev")

	train_output = read_output(args.train_action)
	dev_output = read_output(args.dev_action)
	train_action, actn_v = output2action(train_output, actn_v)
	dev_actoin, actn_v = output2action(dev_output, actn_v)

	print "word vocabulary size:", word_v.size()
	print "char vocabulary size:", char_v.size() - 1
	print "pretrain vocabulary size:", pretrain.size() - 1
	extra_vl_size = []
	for i in range(len(extra_vl)):
		print "extra", i, "vocabulary size:", extra_vl[i].size()
		extra_vl_size.append(extra_vl[i].size())
	print "action vocaluary size:", actn_v.size() - 1
	actn_v.freeze()

	# neural components
	input_representation = token_representation(word_v.size(), char_v.size(), pretrain, extra_vl_size, args)
	encoder = None
	if args.encoder == "BILSTM":
		encoder = bilstm_encoder(args)
	elif args.encoder == "Transformer":
		encoder = transformer(args)
	assert encoder, "please specify encoder type"
	
	decoder = in_order_constituent_parser(actn_v.size(), args)
	mask = in_order_constituent_parser_mask(actn_v)

	encoder_optimizer = optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()), lr=args.learning_rate_f, weight_decay=args.weight_decay_f)
    decoder_optimizer = optim.Adam(filter(lambda p: p.requires_grad, decoder.parameters()), lr=args.learning_rate_f, weight_decay=args.weight_decay_f)
	#training process
	for i, (instance, action) in enumerate(zip(train_instance, train_action)):
		input_embeddings = input_representation(instance, singleton_idx_dict, test=False)
		enc_rep = encoder(input_embeddings)
		loss, _ = decoder(action, mask, enc_rep, test=False)
		loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

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
	subparser.add_argument("--dev-action", default="data/22.action")
	subparser.add_argument("--batch-size", type=int, default=250)
	subparser.add_argument("--checks-per-epoch", type=int, default=4)
	subparser.add_argument("--encoder", default="BILSTM", help="BILSTM, Transformer")
	subparser.add_argument("--use-char", action='store_true')
	subparser.add_argument("--pretrain-path")
	subparser.add_argument("--gpu", type=bool, default=use_cuda)

	args = parser.parse_args()
	args.callback(args)
