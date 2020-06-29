
def get_dummy_corpus(size='small'):
	if size == 'small':
		corpus = [['hola', ',', 'esto', 'es', 'una', 'prueba'],
				  ['esto', 'es', 'otra', 'prueba'],
				  ['esta', 'es', 'Una', 'prueba', 'más']]
	elif size == 'medium':
		corpus = [('w1 w2 w3 w2 w2 w3 w1 w4 w5 w5'*100).split(' '),
				  ('w1 w2 w3 w1 w4 w3 w1 w4 w5 w5'*50).split(' '),
				  ('w1 w2 w0 w2 w2 w2 w2 w4 w5 w5'*1000).split(' ')]
	elif size == 'big':
		corpus = [('w1 w2 w3 w2 w2 w3 w1 w4 w5 w5'*100).split(' '),
				  ('w1 w2 w3 w1 w4 w3 w1 w4 w5 w5'*500).split(' '),
				  ('w1 w2 w0 w2 w2 w2 w2 w4 w5 w5'*1000).split(' ')] * 100
	return corpus

