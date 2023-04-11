
import numpy as np
import re
class Preprocessing:
	
	@staticmethod
	def read_dataset(file):
		
		letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m',
					'n','o','p','q','r','s','t','u','v','w','x','y','z',' ']
		
		text = []
		with open(file, 'r') as f:
			for line in f:
				line = line.lower()
				line = re.sub(r'\d+:', ' ', line)
				line = re.sub(r'\s+', ' ', line)
				line = line.strip()
				if line == '':
					continue
				text += line.split(' ')
				text += ['\n']
	
		return text
		
	@staticmethod
	def create_dictionary(text):
		
		char_to_idx = dict()
		idx_to_char = dict()

		idx = 0
		for char in text:
			if char not in char_to_idx.keys():
				char_to_idx[char] = idx
				idx_to_char[idx] = char
				idx += 1
				
		print("Vocab: ", len(char_to_idx))
		
		return char_to_idx, idx_to_char
		
	@staticmethod
	def build_sequences_target(text, char_to_idx, window):
		
		x = list()
		y = list()
	
		for i in range(len(text)):
			try:
				# Get window of chars from text
				# Then, transform it into its idx representation
				sequence = text[i:i+window]
				sequence = [char_to_idx[char] for char in sequence]
				
				# Get char target
				# Then, transfrom it into its idx representation
				target = text[i+window]
				target = char_to_idx[target]
				
				# Save sequences and targets
				x.append(sequence)
				y.append(target)
			except:
				pass
		
		x = np.array(x)
		y = np.array(y)
		
		return x, y
