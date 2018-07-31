# -*- coding: utf-8 -*-
def read_token_sequence(filename):
	data = []
	with open(filename, "r") as r:
		while True:
			l = r.readline().strip()
			if not l:
				break
			data.append(l)
	return data



