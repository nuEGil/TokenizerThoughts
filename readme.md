# Rough Draft of Tokenization background
Check out the ipython notebook for how to make and run the c++ code. It also has calls for the c++ programs and the python scripts. 

# Abstract
Transformers and scaling seems to be everything. There's a systems level approach to the optimization probelm that involves the tokenizer step. so scale, but scale efficiently. 
---

# Contents
1. Floating point operation and precision 
2. Transformers and scaling
3. Tokenizers
4. Probability perspective
5. Takeaways
6. References

# Floating point operation and precision
In 1991 David Goldberg published a paper on floating point operations and resulting error propagation in computation [1]. His goal was to inform a discussion on the IEEE floating point standard, and provide some rationale for building better / standardized floating point support into computer systems in general. Floating point operations and precision are critical in physics based modeling, agent based simulations, and in control systems work. That means solving differential equations involved in compressible fluids mechanics, and structural mechanics for software involved in designing  aircraft components and in flight contorl systems. It's the same area of work as computing the stress tensors, and strains on a bridge or a skyscraper. For other computational methods, you'd have prediction of disease epidemiology with agent based models, plant optimization software for designing chemical plants and elecritcal grids. In telecomms it was like tracking satelites and sending data back and forth for personal communication. Your buzzwords here, are going to be agent based modeling, finite element analysis, and physics based modeling; Things of that nature. 

This type of software often involves large scale itterative computations. A lot of the math involves vectors, matrices --> tensor operations as the generalized form. CPUs are designed for general purpose and sequential operations, but tensor math can be heavily parallelized. GPUs are designed with that in mind. NVIDIA ends up releasing CUDA in 2007 as a way to give developers a general programming interface for GPUs, and suddenly you can start running larger computations. This is great for applications that have analytic solutions or that have numerically acheivable solutions. But in cases like image classification and language analysis your best bet if general function approximation. There's a proof showing that neural networks are universal function approximators, but the challenge is achiving a useful approximation given a finite discrete computer. 

The point that Goldberg had brought up is that youd end up with large errors in numerical approaches because of lack of guard bits, rounding, and substration based error. As a quick example of something like this, you can take a look at the test function optimization page on wikipedia [8]. These functions are notoriously difficult to numerically optimize, for a variety of reasons. Some have multiple global minima, lots of local minima, exhibit saddle points, and can have different optima depending on the dimension of the input. In some cases you need a high degree of numerical precision to even be able to search the surface. If you're designing any kind of optimizer algorithm like gradient descent and its variants then these are the test functions that show where that method will be effective. For loss functions we can assume a particular manifold, but in practice that surface depends on the data and the model being used; meaning that we will run into variations of these optimizer problems over the course of designing machine learning applications.  It's why in 2014 we get Adam, an updated gradient descent with momentum [18]. Then in 2017 there's AdamW with the decoupled weight decay variant[9]. And the improvement continues. 


# Transformers and scaling
Now with all that in mind, in 2012 Kischevsky and Sutskever under Hinton train AlexNet on 2 NVIDIA GPUs and win a large scale image reconition challenge kicking off a new deep learning boom. 

* 2022 - FP8 formats [6] -- now we are back to gold berg, since we are concerned about classifiers we can actually get rid of a alot of the floating point precision in the parameters -- keep it int the optimizer and in the gradients, and get models that still train well. More on model quantization [7].

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
Pool all the text you can -- you have a higher probability of seeing common words and short words than you do of seeing key words. This is especially the case when the corpus includes multiple domains. So data on Dostoevsky is going to have a different distribution than data on malaria -- and malaria data is going to change from the 1900s to today with the introduction of novel medical technology used in its study. What happens when the language updates? field changes over time. ... recent shift in industry folks only thinking of LLMs as AI despite all the other algorithms out there. -- I think Lanier and Wiener have quotes on this topic, but I need to check . Either way it implies the need for a continuously updating system.  This is what it means to detect distribution shift ... hm. 

Wikipedia has a frequency list for every language ... and for English it has the words on wikipedia -- oh. run the tokenizer on the ICD10 based pages, vs the word list...[]

We know this - it's why TFIDF exists. But it's still something to consider in the tokinzation step. The token set you learn informs the embeddings which inform the transformer which returns novel token sequences --> or if you're classifying something else then it's classifying right .

# Takeaways
1. Deep learning has always been about a numerical solution to an analyitic approximation problem. We know we can approximate any function because theres a proof for that, but now train it...
   
2. Assuming that the scaling thing is just a property of networks, then if we get more efficient sub-blocks and scale those, then we should be able to get better models down the line.... This got me thiking about the tokenizer -> and now at the end of this short rabbit hole -- ended up seeing that tokenizers really act to compress the input in a meaningful way before you ever talk about training an embedding --> plus , from a probabilty stand point you should get different token sets on different data sub batches than you would 

3. For specialized domains we care about the tokenizer, dont just use the techniques right out of the box. - maybe. 

4. What happens when the language updates? New protein is discovered, people change the definition and usage of words over time, etc.  

5. On pricing that you see on most API providers. Watts and Volts are fixed, tokens are not. Would be better to do measures in floating point operations, or bytes or something.

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

[18] Diederik Kingma, Jimmy Ba. Adam: A method for stochastic optimization (22 Dec 2014) https://arxiv.org/abs/1412.6980
