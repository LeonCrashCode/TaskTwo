
class vocabulary:
	def __init__(self):
		self._tti = {"<UNK>":0}
		self._itt = ["<UNK>"]
		self._frozen = False
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
	def toidx(self, tok):
		if tok in self._tti:
			return self._tti[tok]

		if self._frozen == False:
			self._tti[tok] = len(self._itt)
			self._itt.append(tok)
			return len(self._itt) - 1
		else:
			return 0
	def totok(self, idx):
		assert idx < self.size, "Out of Vocabulary"
		return self._itt[idx]
	def size(self):
		return len(self._tti)

class PretrainedEmb:
	def __init__(self, filename):
		self._ttv = [[]] #leave a space for UNK
		self._v = vocabulary()
		self.read_file(filename)
	def read_file(self, filename):
		if not filename:
			return
		with open(filename, "r") as r:
			while True:
				l = r.readline().strip()
				if not l:
					break
				l = l.split()
				idx = self._v.toidx(l[0])
				assert idx == len(self._ttv)
				self._ttv.append([float(t) for t in l[1:]])
				if len(self._ttv[0]) == 0:
					self._ttv[0] = [ 0.0 for i in range(len(l)-1)]
				for i in range(len(self._ttv[0])):
					self._ttv[0][i] += float(l[i+1])
		for i in range(len(self._ttv[0])):
			self._ttv[0][i] /= (len(self._ttv) - 1)
		self._v.freeze()
	def toidx(self, tok):
		return self._v.toidx(tok)
	def size(self):
		return self._v.size()
	def vectors(self):
		return self._ttv