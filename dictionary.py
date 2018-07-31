
class vocabulary:
	def __init__(self):
		_tti = {"<UNK>":0}
		_itt = ["<UNK>"]
		_frozen = False
	def freeze(self):
		self._frozen = True
	def unfreeze(self):
		self._frozen = False
	def read_file(self, filename):
		with open(filename, "r") as r:
			while True:
				l = r.readline().strip()
				if not l:
					break
				self.toidx(l)
		self.size = len(self._itt)
	def toidx(self, tok):
		if tok in self._tti:
			return self._tti[tok]

		if self._forzen == False:
			self._tti[tok] = len(self._itt)
			self._itt.append(tok)
			return len(self._itt) - 1
		else:
			return 0
	def totok(self, idx):
		assert idx < self.size, "Out of Vocabulary"
		return self._itt[idx]

class PretraiedEmb:
	def __init__(self):
		_ttv = {}
	def read_file(self, filename):
		with open(filename, "r") as r:
			while True:
				l = r.readline().strip()
				if not l:
					break
				l = l.split()
				self._ttv[l[0]] = [ float(t) for t in l.split()[1:]]

