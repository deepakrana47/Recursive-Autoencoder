# Recursive-Autoencoder
Consist of Recursive Autoencoder for phrase and sentence vector generation using word vectors and dependency tree

To execute the code python 2.7 and following packages are required:

> keras

> numpy

> practnlptools

To perform paraphrasing detection test run the following:

> $ python sent_test.py
  
E.g.:

> $ python sent_test.py

> Enter Sentence1 : Consumers would still have to get a descrambling security card from their cable operator to plug into the set.

> Enter Sentence2 : To watch pay television, consumers would insert into the set a security card provided by their cable service.


> Result:  Not Paraphrase


> Phrase Similarity value(0=similar, above 0=dissimilarity index)

> 	('consumers', 'cable') :		0.824109976887

> 	('still', 'set') :		0.680742255176

> 	('descrambling', 'cable') :		0.944514994194

> 	('security', 'security') :		0.0

> 	('cable', 'cable') :		0.0

> 	('set', 'set') :		0.0

> 	('plug set', 'the set') :		0.800121098315

> 	('cable operator', 'cable service') :		0.765510384122

> 	('descrambling security card', 'television consumers') :		1.01668589084

> 	('get descrambling security card cable operator plug set', 'pay television consumers') :		1.17636987178

> 	('consumers still have get descrambling security card cable operator plug set', 'watch pay television consumers') :		1.1848175183
