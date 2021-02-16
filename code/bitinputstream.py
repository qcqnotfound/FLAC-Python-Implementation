class BitInputStream(object):
	
	def __init__(self, inp):
		self.inp = inp
		self.bitbuffer = 0
		self.bitbufferlen = 0
	
	
	def align_to_byte(self):
		self.bitbufferlen -= self.bitbufferlen % 8
	
	
	def read_byte(self):
		if self.bitbufferlen >= 8:
			return self.read_uint(8)
		else:
			result = self.inp.read(1)
			if len(result) == 0:
				return -1
			return result[0]
	
	
	def read_uint(self, n):
		while self.bitbufferlen < n:
			temp = self.inp.read(1)
			if len(temp) == 0:
				raise EOFError()
			temp = temp[0]
			self.bitbuffer = (self.bitbuffer << 8) | temp
			self.bitbufferlen += 8
		self.bitbufferlen -= n
		result = (self.bitbuffer >> self.bitbufferlen) & ((1 << n) - 1)
		self.bitbuffer &= (1 << self.bitbufferlen) - 1
		return result
	
	
	def read_signed_int(self, n):
		temp = self.read_uint(n)
		temp -= (temp >> (n - 1)) << n
		return temp
	
	
	def read_rice_signed_int(self, param):
		val = 0
		while self.read_uint(1) == 0:
			val += 1
		val = (val << param) | self.read_uint(param)
		return (val >> 1) ^ -(val & 1)
	
	
	def close(self):
		self.inp.close()
	
	
	def __enter__(self):
		return self
	
	
	def __exit__(self, type, value, traceback):
		self.close()