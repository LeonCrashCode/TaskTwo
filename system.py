import torch

def system_check_and_init(args):
	if args.gpu:
		assert torch.cuda.is_available(), "GPU is not available."
	if args.gpu:
		torch.cuda.manual_seed_all(12345678)
	torch.manual_seed(12345678)