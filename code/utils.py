import numpy as np
import torch



def print_data_info(train_data, valid_data, test_data, src_field, trg_field):
	""" This prints some useful stuff about our data sets. """

	print("Data set sizes (number of sentence pairs):")
	print('train', len(train_data))
	print('valid', len(valid_data))
	print('test', len(test_data), "\n")

	print("First training example:")
	print("src:", " ".join(vars(train_data[0])['src']))
	print("trg:", " ".join(vars(train_data[0])['trg']), "\n")

	print("Most common words (src):")
	print("\n".join(["%10s %10d" % x for x in src_field.vocab.freqs.most_common(10)]), "\n")
	print("Most common words (trg):")
	print("\n".join(["%10s %10d" % x for x in trg_field.vocab.freqs.most_common(10)]), "\n")

	print("First 10 words (src):")
	print("\n".join(
		'%02d %s' % (i, t) for i, t in enumerate(src_field.vocab.itos[:10])), "\n")
	print("First 10 words (trg):")
	print("\n".join(
		'%02d %s' % (i, t) for i, t in enumerate(trg_field.vocab.itos[:10])), "\n")

	print("Number of German words (types):", len(src_field.vocab))
	print("Number of English words (types):", len(trg_field.vocab), "\n")



def greedy_decode(model, src, src_mask, src_lengths, max_len=100, sos_index=1, eos_index=None):
	"""Greedily decode a sentence."""

	with torch.no_grad():
		encoder_hidden, encoder_final = model.encode(src, src_mask, src_lengths)
		prev_y = torch.ones(1, 1).fill_(sos_index).type_as(src)
		trg_mask = torch.ones_like(prev_y)

	output = []
	attention_scores = []
	hidden = None

	for i in range(max_len):
		with torch.no_grad():
			out, hidden, pre_output = model.decode(
			  encoder_hidden, encoder_final, src_mask,
			  prev_y, trg_mask, hidden)

			# we predict from the pre-output layer, which is
			# a combination of Decoder state, prev emb, and context
			prob = model.generator(pre_output[:, -1])

		_, next_word = torch.max(prob, dim=1)
		next_word = next_word.data.item()
		output.append(next_word)
		prev_y = torch.ones(1, 1).type_as(src).fill_(next_word)
		attention_scores.append(model.decoder.attention.alphas.cpu().numpy())
	
	output = np.array(output)
		
	# cut off everything starting from </s> 
	# (only when eos_index provided)
	if eos_index is not None:
		first_eos = np.where(output==eos_index)[0]
		if len(first_eos) > 0:
			output = output[:first_eos[0]]
	
	return output, np.concatenate(attention_scores, axis=1)
  

def lookup_words(x, vocab=None):
	if vocab is not None:
		x = [vocab.itos[i] for i in x]

	return [str(t) for t in x]





def print_examples(example_iter, model, n=2, max_len=100, 
				   sos_index=1, 
				   src_eos_index=None, 
				   trg_eos_index=None, 
				   src_vocab=None, 
				   trg_vocab=None):
	"""Prints N examples. Assumes batch size of 1."""

	SOS_TOKEN = "<s>"
	EOS_TOKEN = "</s>"
	model.eval()
	count = 0
	print()
	
	if src_vocab is not None and trg_vocab is not None:
		src_eos_index = src_vocab.stoi[EOS_TOKEN]
		trg_sos_index = trg_vocab.stoi[SOS_TOKEN]
		trg_eos_index = trg_vocab.stoi[EOS_TOKEN]
	else:
		src_eos_index = None
		trg_sos_index = 1
		trg_eos_index = None
		
	for i, batch in enumerate(example_iter):
	  
		src = batch.src.cpu().numpy()[0, :]
		trg = batch.trg_y.cpu().numpy()[0, :]

		# remove </s> (if it is there)
		src = src[:-1] if src[-1] == src_eos_index else src
		trg = trg[:-1] if trg[-1] == trg_eos_index else trg      
	  
		result, _ = greedy_decode(
		  model, batch.src, batch.src_mask, batch.src_lengths,
		  max_len=max_len, sos_index=trg_sos_index, eos_index=trg_eos_index)
		print("Example #%d" % (i+1))
		print("Src : ", " ".join(lookup_words(src, vocab=src_vocab)))
		print("Trg : ", " ".join(lookup_words(trg, vocab=trg_vocab)))
		print("Pred: ", " ".join(lookup_words(result, vocab=trg_vocab)))
		print()
		
		count += 1
		if count == n:
			break







