import argparse
from dictionary import vocabulary

def run_train(args, hypers):
	#read training instances
	with open(args.)

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

	args = parser.parse_args()
	args.callback(args)
