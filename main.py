import argparse
from dictionary import vocabulary
from dictionary import PretrainedEmb
from utils import read_input
from utils import get_singleton_dict
from utils import input2instance
from representation import token_representation

def run_train(args, hypers):
	word_v = vocabulary()
	char_v = vocabulary()
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

	print "word vocabulary size:", word_v.size()
	print "char vocabulary size:", char_v.size() - 1
	print "pretrain vocabulary size:", pretrain.size() - 1
	extra_vl_size = []
	for i in range(len(extra_vl)):
		print "extra", i, "vocabulary size:", extra_vl[i].size()
		extra_vl_size.append(extra_v[i].size())

	input_representation = token_representation(word_v.size(), char_v.size(), pretrain, extra_vl_size, args)
	for instance in train_instance:
		input_representation(instance, singleton_idx_dict, test=False, gpu=False)
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	subparsers = parser.add_subparsers()

	hypers = {}
	for line in open("config"):
		line = line.strip()
		if not line:
			break
		hypers[line.split()[0]] = line.split()[1]

	subparser = subparsers.add_parser("train")
	subparser.set_defaults(callback=lambda args: run_train(args, hypers))
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

	args = parser.parse_args()
	args.callback(args)
