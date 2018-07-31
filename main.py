import argparse
from dictionary import vocabulary
from dictionary import PretrainedEmb
from io import read_input
def run_train(args, hypers):
	#read training instances
	word_v = vocabulary()
	char_v = vocabulary()
	pretrain = PretrainedEmb(args.pretrain_path)

	train_input = read_input(args.train_input)
	#lexicon representation
	train_instance = [[]]
	for instance in train_input:
		instance[-1].append([]) # word
		instance[-1].append([]) # char
		instance[-1].append([]) # pretrain
		for w in instance[0]:
			idx = word_v.toidx(w)
			instance[-1][-3].append(idx)
			if args.use_char:
				instance[-1][-2].append([])
				for c in list(w):
					idx = char_v.toidx(c)
					instance[-1][-2][-1].append(idx)
			if args.pretrain_path:
				idx = pretran.toidx(w.lower())
				instance[-1][-2].append(idx)

	dev_input = read_input(args.dev_input)


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
	subparser.add_argument("--train-input", default="data/train.input")
	subparser.add_argument("--train-action", default="data/train.action")
	subparser.add_argument("--dev-input", default="data/dev.input")
	subparser.add_argument("--dev-action", default="data/dev.action")
	subparser.add_argument("--batch-size", type=int, default=250)
	subparser.add_argument("--checks-per-epoch", type=int, default=4)
	subparser.add_argument("--encoder", default="BILSTM", help="BILSTM, Transformer")
	subparser.add_argument("--use-char", type=bool)
	subparser.add_argument("--pretrain-path")

	args = parser.parse_args()
	args.callback(args)
