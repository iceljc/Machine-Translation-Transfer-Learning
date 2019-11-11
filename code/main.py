import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchtext import data, datasets
from data import rebatch
from model import make_model, SimpleLossCompute
from utils import print_data_info, print_examples, greedy_decode, lookup_words
from settings import params
import sacrebleu


def run_epoch(data_iter, model, loss_compute, print_every=50):
	"""Standard Training and Logging Function"""

	start = time.time()
	total_tokens = 0
	total_loss = 0
	print_tokens = 0

	for i, batch in enumerate(data_iter, 1):
		
		out, _, pre_output = model.forward(batch.src, batch.trg,
										   batch.src_mask, batch.trg_mask,
										   batch.src_lengths, batch.trg_lengths)
		loss = loss_compute(pre_output, batch.trg_y, batch.nseqs)
		total_loss += loss
		total_tokens += batch.ntokens
		print_tokens += batch.ntokens
		
		if model.training and i % print_every == 0:
			elapsed = time.time() - start
			print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
					(i, loss / batch.nseqs, print_tokens / elapsed))
			start = time.time()
			print_tokens = 0

	return math.exp(total_loss / float(total_tokens))


def train(model, train_iter, valid_iter, num_epochs=10, lr=0.0003, print_every=100):
	"""Train a model on IWSLT"""
	
	if params['USE_CUDA']:
		model.cuda()

	patience = 5
	patience_count = 0
	best = 1e10

	# optionally add label smoothing; see the Annotated Transformer
	criterion = nn.NLLLoss(reduction="sum", ignore_index=PAD_INDEX)
	optim = torch.optim.Adam(model.parameters(), lr=lr)
	
	train_perplexities, dev_perplexities = [], []

	for epoch in range(1, num_epochs+1):
	  
		print("Epoch", epoch)
		model.train()
		train_perplexity = run_epoch((rebatch(PAD_INDEX, b) for b in train_iter), 
									 model,
									 SimpleLossCompute(model.generator, criterion, optim),
									 print_every=print_every)
		print("Training perplexity: %f" % train_perplexity)
		train_perplexities.append(train_perplexity)
		model.eval()
		with torch.no_grad():
			# print_examples((rebatch(PAD_INDEX, x) for x in valid_iter), 
			# 			   model, n=3, src_vocab=SRC.vocab, trg_vocab=TRG.vocab)        

			dev_perplexity = run_epoch((rebatch(PAD_INDEX, b) for b in valid_iter), 
									   model, 
									   SimpleLossCompute(model.generator, criterion, None))
			print("Validation perplexity: %f" % dev_perplexity)
			dev_perplexities.append(dev_perplexity)

		# early stoping
		is_best = dev_perplexity < best
		best = min(dev_perplexity, best)

		if is_best:
			patience_count = 0
		else:
			patience_count += 1
			if patience_count >= patience:
				print("Early stopping at epoch {}".format(epoch))
				break

	print("Saving the model...")
	with open('mt_model2.pt', 'wb') as f:
		torch.save(model, f)
		
	return train_perplexities, dev_perplexities


def run_test(model, test_data, test_iter):
	preds = []
	
	for k, batch in enumerate(test_iter):
		model.eval()
		with torch.no_grad():
			batch = rebatch(PAD_INDEX, batch)
			pred, attn = greedy_decode(model, batch.src, batch.src_mask, batch.src_lengths, max_len=25, 
				sos_index=TRG.vocab.stoi[SOS_TOKEN], eos_index=TRG.vocab.stoi[EOS_TOKEN])

			preds.append(pred)

	hypotheses = [lookup_words(pred, TRG.vocab) for pred in preds]
	hypotheses = [" ".join(h) for h in hypotheses]
	references = [" ".join(data.trg) for data in test_data]
	bleu = sacrebleu.raw_corpus_bleu(hypotheses, [references], .01).score
	print("BLEU score: ", bleu)

	return bleu


if __name__ == '__main__':

	seed = 42
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	DEVICE = params['DEVICE']

	# import data
	import spacy
	spacy_de = spacy.load('de')
	spacy_en = spacy.load('en')

	def tokenize_de(text):
		return [tok.text for tok in spacy_de.tokenizer(text)]

	def tokenize_en(text):
		return [tok.text for tok in spacy_en.tokenizer(text)]

	UNK_TOKEN = "<unk>"
	PAD_TOKEN = "<pad>"    
	SOS_TOKEN = "<s>"
	EOS_TOKEN = "</s>"
	LOWER = True

	# we include lengths to provide to the RNNs
	SRC = data.Field(tokenize=tokenize_en, 
					 batch_first=True, lower=LOWER, include_lengths=True,
					 unk_token=UNK_TOKEN, pad_token=PAD_TOKEN, init_token=None, eos_token=EOS_TOKEN)
	TRG = data.Field(tokenize=tokenize_de, 
					 batch_first=True, lower=LOWER, include_lengths=True,
					 unk_token=UNK_TOKEN, pad_token=PAD_TOKEN, init_token=SOS_TOKEN, eos_token=EOS_TOKEN)

	# data split
	MAX_LEN = 25  # NOTE: we filter out a lot of sentences for speed
	train_data, valid_data, test_data = datasets.IWSLT.splits(
		exts=('.en', '.de'), fields=(SRC, TRG), 
		filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and 
			len(vars(x)['trg']) <= MAX_LEN)

	MIN_FREQ = 5  # NOTE: we limit the vocabulary to frequent words for speed
	SRC.build_vocab(train_data.src, min_freq=MIN_FREQ)
	TRG.build_vocab(train_data.trg, min_freq=MIN_FREQ)

	PAD_INDEX = TRG.vocab.stoi[PAD_TOKEN]

	# print data info
	print_data_info(train_data, valid_data, test_data, SRC, TRG)

	# define iterator
	train_iter = data.BucketIterator(train_data, batch_size=params['batch_size'], train=True, 
								 sort_within_batch=True, 
								 sort_key=lambda x: (len(x.src), len(x.trg)), repeat=False,
								 device=DEVICE)
	valid_iter = data.Iterator(valid_data, batch_size=1, train=False, sort=False, repeat=False, 
							device=DEVICE)

	test_iter = data.Iterator(test_data, batch_size=1, train=False, sort=False, repeat=False, device=DEVICE)

	print("Building the model ...")
	saved_model = 'mt_model.pt'
	model = make_model(SRC.vocab, len(SRC.vocab), len(TRG.vocab), emb_size=params['embed_size'], 
		hidden_size=params['hidden_size'], num_layers=params['num_layers'], dropout=params['dropout'], resume=saved_model)
	
	print("Training the model ...")
	train_perplexities, dev_perplexities = train(model, train_iter, valid_iter, num_epochs=params['num_epochs'], 
		lr=params['learning_rate'], print_every=100)

	print("Testing the model ...")
	bleu_score = run_test(model, test_data, test_iter)


	# save data
	with open("result/train_loss2.txt", "w") as f:
		for item in train_perplexities:
			f.write("%s\n" % item)

	with open("result/valid_loss2.txt", "w") as f:
		for item in dev_perplexities:
			f.write("%s\n" % item)

















