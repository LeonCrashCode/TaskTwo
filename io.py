# -*- coding: utf-8 -*-
def read_input(filename):
	data = [[]]
	for line in open(filename):
		line = line.strip()
		if line == "":
			data.append([])
		else:
			data[-1].append(line.split())
	return data



