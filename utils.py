# -*- coding: utf-8 -*-
def read_input(filename):
	data = [[]]
	for line in open(filename):
		line = line.strip()
		if line == "":
			data.append([])
		else:
			data[-1].append(["<s>"] + line.split() + ["</s>"])
	if len(data[-1]) == 0:
		data.pop()
	return data

def get_singleton_dict(train_input, word_v):
	d = {}
	singleton_idx_dict = {}
	word_dict = {}
	for instance in train_input:
		for w in instance[0]:
			if w in d:
				d[w] += 1
			else:
				d[w] = 1
			word_v.toidx(w)

	for key in d.keys():
		if d[key] == 1:
			singleton_idx_dict[word_v.toidx(key)] = 1
		else:
			word_dict[key] = 1
	return singleton_idx_dict, word_dict, word_v

def unkify(token, word_dict, lang):
	if len(token.rstrip()) == 0:
		return '<UNK>'
	elif not(token.rstrip() in word_dict):
		if lang == "ch":
			return '<UNK>'
		numCaps = 0
		hasDigit = False
		hasDash = False
		hasLower = False
		for char in token.rstrip():
			if char.isdigit():
				hasDigit = True
			elif char == '-':
				hasDash = True
			elif char.isalpha():
				if char.islower():
					hasLower = True
				elif char.isupper():
					numCaps += 1
		result = '<UNK>'
		lower = token.rstrip().lower()
		ch0 = token.rstrip()[0]
		if ch0.isupper():
			if numCaps == 1:
				result = result + '-INITC'
				if lower in word_dict:
					result = result + '-KNOWNLC'
			else:
				result = result + '-CAPS'
		elif not(ch0.isalpha()) and numCaps > 0:
			result = result + '-CAPS'
		elif hasLower:
			result = result + '-LC'
		if hasDigit:
			result = result + '-NUM'
		if hasDash:
			result = result + '-DASH'
		if lower[-1] == 's' and len(lower) >= 3:
			ch2 = lower[-2]
			if not(ch2 == 's') and not(ch2 == 'i') and not(ch2 == 'u'):
				result = result + '-s'
		elif len(lower) >= 5 and not(hasDash) and not(hasDigit and numCaps > 0):
			if lower[-2:] == 'ed':
				result = result + '-ed'
			elif lower[-3:] == 'ing':
				result = result + '-ing'
			elif lower[-3:] == 'ion':
				result = result + '-ion'
			elif lower[-2:] == 'er':
				result = result + '-er'
			elif lower[-3:] == 'est':
				result = result + '-est'
			elif lower[-2:] == 'ly':
				result = result + '-ly'
			elif lower[-3:] == 'ity':
				result = result + '-ity'
			elif lower[-1] == 'y':
				result = result + '-y'
			elif lower[-2:] == 'al':
				result = result + '-al'
		return result
	else:
		return token.rstrip() 

def input2instance(train_input, word_v, char_v, pretrain, extra_vl, word_dict, args, op):
	train_instance = []
	for instance in train_input:
		#lexicon representation
		train_instance.append([])
		train_instance[-1].append([]) # word
		train_instance[-1].append([]) # char
		train_instance[-1].append([]) # pretrain
		train_instance[-1].append([]) # typed UNK
		for w in instance[0]:
			idx = word_v.toidx(w)
			if op == "train":
				train_instance[-1][0].append(idx)
				idx = word_v.toidx(unkify(w, word_dict, "en"))
				train_instance[-1][3].append(idx)
			elif op == "dev":
				if idx == 0: # UNK
					idx = word_v.toidx(unkify(w, word_dict, "en", force=True))
				train_instance[-1][0].append(idx)

			if args.use_char:
				train_instance[-1][1].append([])
				for c in list(w):
					idx = char_v.toidx(c)
					train_instance[-1][1][-1].append(idx)
			if args.pretrain_path:
				idx = pretrain.toidx(w.lower())
				train_instance[-1][2].append(idx)
		#extra representatoin
		for i, extra_info in enumerate(instance[1:], 0):
			train_instance[-1].append([])
			for t in extra_info:
				idx = extra_vl[i].toidx(t)
				train_instance[-1][-1].append(idx)

	return train_instance, word_v, char_v, extra_vl

def read_output(filename):
	data = []
	for line in open(filename):
		line = line.strip()
		data.append(line.split())
	return data

def output2action(train_output, actn_v):
	train_action = []
	for output in train_output:
		train_action.append([])
		for a in output:
			idx = actn_v.toidx(a)
			train_action[-1].append(idx)
	return train_action, actn_v

