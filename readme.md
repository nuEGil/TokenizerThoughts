# Rough Draft of Tokenization background

* still need to make a step to ingest the learned token sets and use that in a python script to run the tokenization.... 
* also need to try to compare token sets across different books... 
* and then do a histogram of the sequence legnths.... 
in progress. 

On pricing that you see on most API providers. Watts and Volts are fixed, tokens are not. Would be better to do measures in floating point operations, or bytes or something. 

# Abstract
Transformers and scaling seems to be everything. There's a systems level approach to the optimization probelm that involves the tokenizer step. so scale, but scale efficiently. --

# Contents
1. Floating point operation and precision. 
2. Transformers and scaling
3. Tokenizers
4. Probability perspective
5. Takeaways
6. References

# Floating point operation and precision
* Goldberg published a paper on the IEEE floating point standard to provide better support for floating point [1]. It's 1991 and it's a while before AlexNet kicks off a DL renaissance. Computational modeling is more concerned with physics based simulations. You could think about solving differential equations involved in compressible fluids mechanics, and structural mechanics for things like designing novel aircraft wings or for better flight controls systems. It could be computational software to compute the shear forces on a bridge or a sky scraper -- things of this nature.
* Wikipedia on test functions for optimization problems - this gives some intuition and test cases for whatever your optimizer is and whether your loss function is learnable.[8]. Pick a function, we will see convexity, high local optima, and saddle points. Then things change as these functions get into higher dimenssions. Any loss function you write for a deep learning problem is gonna run into these types of issues in high dimensions.  
* AdamW - Adam was adaptive momentum, you can decouplpe the weight decay regularization...[9]
* 2022 - FP8 formats [6] -- now we are back to gold berg, since we are concerned about classifiers we can actually get rid of a alot of the floating point precision in the parameters -- keep it int the optimizer and in the gradients, and get models that still train well. More on model quantization [7].

# Transformers and scaling
* 2017 we get the transformer achitecture in attention is all you need [2]. Offers more representational capacity than we have from previous models --> look at the math behind the attention block (after the dash is a rabit hole for that brunton section)- looks like SVD -- so a learnable kernel svd basically. SVD is just one of many bases, so this gets
* 2019 Deep double descent [3]-- hyper scaling starts as we see a regime where model testing loss gets way better than we thought - originally we thought the models would just overfit. this is apparently not the case. -- demonstrated for resnet and transformer - i think, need to check 
* 2020 Scaling Laws for Neural Language models [4]- we can predictably see that the models will get logarithmic scaling with more data, more parameters, and more training... so push that till we can't  -- seems like the mesage. 
* 2022 - Flash attention [5] ok we know that we are going to scale, but can we be more efficient about it?

* SVD and compressed sensing? Transformers as compression? Tokenizer is definitely compression.. 

# Tokenizers
* A tokenizer takes a small chunk of bytes and says - that's number 1, or that's 233. It already has some element of compression involved --> rewrite 4 bytes as 1 in a look up table.
* BPE already is derived from a compression algorithm -- bottom up- so start with common bigrams and end with sequences. WordPiece is also bottom up. Unigram is top down. 
* BERT and versions of BERT all of them use WordPeice - BERT Original paper used WordPiece Tokenization [10] ( see how much data it was trained on ). BioBert [11] - custom version of bert using - 4.5B words, 13.5B words. PsychBert -pay wall [12]
* Medical Knowledge repre [13] -- uses byte pair encoding BECAUSE it will give you custom text
* Comes up a lot in other languages  --turkish model [14] already doing this comparison between WorPiece and BPE.

...we need the array implementation to work on large corpora -> 1.7MB @ 5s --> 3.4 million tokens per second is not good enough hm . actually. at 1.47billion (e9) words... say...x3 - looking at 1297s or about a half an hour to tokenize something like 18GB of text.... but I havent evaluated this algorithm for long runs -- if average step time changes over that 30 minute run, then there's a memory ineficiency... hm. also havent tried writing a python or a bash script to call like 5 of those execuatbles up in parallel... that's something to test out... run in parrallel, consolidate lists at the end. interesting. 

Would be cool to do an array version. But the other thing we need to run this on a few documents and see what the model keeps as far as unique toknes and what it keeps as far as common tokens.   

# Proability perspective
Pool all the text you can -- you have a higher probability of seeing common words and short words than you do of seeing key words. This is especially the case when the corpus includes multiple domains. So data on Dostoevsky is going to have a different distribution than data on malaria -- and malaria data is going to change from the 1900s to today with the introduction of novel medical technology used in its study. 

Wikipedia has a frequency list for every language ... and for English it has the words on wikipedia -- oh. run the tokenizer on the ICD10 based pages, vs the word list...[]

We know this - it's why TFIDF exists. But it's still something to consider in the tokinzation step. The token set you learn informs the embeddings which inform the transformer which returns novel token sequences --> or if you're classifying something else then it's classifying right .

# Takeaways
 
1. Deep learning has always been about a numerical solution to an analyitic approximation problem. We know we can approximate any function because theres a proof for that, but now train it...
   
2. Assuming that the scaling thing is just a property of networks, then if we get more efficient sub-blocks and scale those, then we should be able to get better models down the line.... This got me thiking about the tokenizer -> and now at the end of this short rabbit hole -- ended up seeing that tokenizers really act to compress the input in a meaningful way before you ever talk about training an embedding --> plus , from a probabilty stand point you should get different token sets on different data sub batches than you would 

3. For specialized domains we care about the tokenizer, dont just use the techniques right out of the box. - maybe. 

# References
[1] David Goldberg. 1991. What every computer scientist should know about floating-point arithmetic. ACM Comput. Surv. 23, 1 (March 1991), 5–48. https://doi.org/10.1145/103162.103163.

[2] Ashish Vaswani et al. 2017, Attention is all you need

[3] Preetum Nakkiran et al. Deep Double Descent: Where Bigger Models and More Data Hurt., 2019, arXiv, (4 December 2019), https://arxiv.org/abs/1912.02292.

[4] Jared Kaplan et al. 2020, Scaling Laws for Neural Language Models. (23 Jan 2020) https://arxiv.org/abs/2001.08361

[5] Tri Dao et al. 2022, FlashAttention: Fast and Memory Efficient Exact Attention with IO-Awareness. arXiv, (27 May 2022), https://arxiv.org/abs/2205.14135.

[6] Paulius Micikevicius et al. 2022, FP8 Formats for Deep Learning. arXiv, (12 September 2022), https://arxiv.org/abs/2209.05433.

[7] https://developer.nvidia.com/blog/model-quantization-concepts-methods-and-why-it-matters/

[8] https://en.wikipedia.org/wiki/Test_functions_for_optimization

[9] Ilya Loshchilov et al. Decoupled Weight Decay regularization, arXiv, (14 Nov 2017) https://arxiv.org/abs/1711.05101 

[10] Jacob Devlin et al. BERT: Pre-training of Deep Bidirectional Transfomers for Language Understanding, 2019, arXiv, (11 October 2018) https://arxiv.org/abs/1810.04805

[11] Jinhyuk Lee et al. BioBERT: a pre-trained biomedical represenatation model for biomedical text mining

[12] V. Vajre et al, PsychBERT: A Mental Health Language Model for Social Media Mental Health Behavioral Analysis, 2021 IEEE International Conference on Bioinformatics and Biomedicine (BIBM), Houston, TX, USA, 2021, pp. 1077-1082, doi: 10.1109/BIBM52615.2021.9669469.

[13] Li, Q., Tong, J., Liu, S. et al. Medical knowledge representation enhancement in large language models through clinical tokens optimization. Sci Rep 16, 6563 (2026). https://doi.org/10.1038/s41598-026-37438-6

[14] Cagri Toraman, Eyup Halit Yilmaz, Furkan Şahi̇nuç, and Oguzhan Ozcelik. 2023. Impact of Tokenization on Language Models: An Analysis for Turkish. ACM Trans. Asian Low-Resour. Lang. Inf. Process. 22, 4, Article 116 (April 2023), 21 pages. https://doi.org/10.1145/3578707

[15] https://en.wikipedia.org/wiki/Most_common_words_in_English

[16] https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists

[17] https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists/English/Wikipedia_(2016)/10001-20000

