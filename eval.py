import os
def constituent_parser_eval(args):
	software = args.eval_path_base+"/evalb"
	param = args.eval_path_base+"/COLLINS.prm"
	goldfile = args.dev_output
	outfile = "tmp/dev.output.tmp"
	scorefile = "tmp/dev.eval.tmp"
	cmd = " ".join([software, "-p", param, goldfile, outfile, ">", scorefile])
	print cmd
	os.system(cmd)

	for line in open(scorefile):
		line = line.strip()
		if line == "":
			continue
		line = line.split()
		if " ".join(line[:2]) == "Bracketing FMeasure":
			return float(line[-1])
	assert False, "No score found"

