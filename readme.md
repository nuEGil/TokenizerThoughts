# Rough Draft of Tokenization background
Check out the ipython notebook for how to make and run the c++ code. It also has calls for the c++ programs and the python scripts. 

# Contents
1. Takeaways
2. Timing results
3. Floating point operation and precision 
4. Transformers 
5. Scaling
6. Tokenizers 
7. References

# Takeaways
1. Deep learning has always been about a numerical solution to an continuous analytical approximation problem. We know we can approximate any function because theres a proof for that, but now train it.
   
2. If we want to keep scaling we should invest in efficient scaling, then we should be able to get better models down the line. But tokenizers compress your text sequences so the availability of more and more informative text sequences is questionable. 

3. For specialized domains we care about custom tokenizers. As a field changes, its language changes implying the need to continuously improve /train tokenizers and all the down stream components as a result.   

5. On pricing that you see on most API providers. Watts and Volts are fixed, tokens are not. Would be better to do measures in floating point operations, or bytes.

# Timing results 
512 steps of learning on crime and punishement - 1.7 million chars / 1.7 MB
* Pyhton doubly linked list ~ 1 minute
* C++ doubly linked list base ~ 11.5s 
* C++ doubly linked list version with -O3 flag ~5.7 seconds
* C++ array version -O3 flag ~1.2 s

# Floating point operation and precision
In 1991 David Goldberg published a paper on floating point operations and resulting error propagation in computation [1]. His goal was to inform a discussion on the IEEE floating point standard, and provide some rationale for building better / standardized floating point support into computer systems in general. Floating point operations and precision are critical in physics based modeling, agent based simulations, and in control systems work. That means solving differential equations involved in compressible fluids mechanics, and structural mechanics for software involved in designing  aircraft components and in flight contorl systems. It's the same area of work as computing the stress tensors, and strains on a bridge or a skyscraper. For other computational methods, you'd have prediction of disease epidemiology with agent based models, plant optimization software for designing chemical plants and elecritcal grids. In telecomms it was like tracking satelites and sending data back and forth for personal communication. Your buzzwords here, are going to be agent based modeling, finite element analysis, and physics based modeling; Things of that nature. 

This type of software often involves large scale itterative computations. A lot of the math involves vectors, matrices --> tensor operations as the generalized form. CPUs are designed for general purpose and sequential operations, but tensor math can be heavily parallelized. GPUs are designed with that in mind. NVIDIA ends up releasing CUDA in 2007 as a way to give developers a general programming interface for GPUs, and suddenly you can start running larger computations. This is great for applications that have analytic solutions or that have numerically acheivable solutions. But in cases like image classification and language analysis your best bet if general function approximation. There's a proof showing that neural networks are universal function approximators, but the challenge is achiving a useful approximation given a finite discrete computer. 

The point that Goldberg had brought up is that youd end up with large errors in numerical approaches because of lack of guard bits, rounding, and substration based error. As a quick example of something like this, you can take a look at the test function optimization page on wikipedia [8]. These functions are notoriously difficult to numerically optimize, for a variety of reasons. Some have multiple global minima, lots of local minima, exhibit saddle points, and can have different optima depending on the dimension of the input. In some cases you need a high degree of numerical precision to even be able to search the surface. If you're designing any kind of optimizer algorithm like gradient descent and its variants then these are the test functions that show where that method will be effective. For loss functions we can assume a particular manifold, but in practice that surface depends on the data and the model being used; meaning that we will run into variations of these optimizer problems over the course of designing machine learning applications.  It's why in 2014 we get Adam, an updated gradient descent with momentum [18]. Then in 2017 there's AdamW with the decoupled weight decay variant[9]. And the improvement continues. 

# Transformers
With all that in mind, in 2012 Kischevsky and Sutskever under Hinton train AlexNet (convolutional network) on 2 NVIDIA GPUs and win a large scale image reconition challenge kicking off a new deep learning boom. We get a lot of architecture improvements from residual networks that include an identity function to make it eaiser to propagate errors back during training, to inception models, to RNNs and CNN-LSTM combintaitons for sequence learning etc. In 2017 we get the transformer achitecture [2]. Offers more representational capacity than we have from previous models.

Here's a handwaving argument for this. The formula for the attention block is

    Attention = Softmax((QK)/sqrt(dk)) V

Q, K, and V expand out to matrices operating on input vectors. So you're applying a kernel on 2 matrix operators scaling the output and then doing one more matrix operation. Compare that to singular value decomposition which is

    x = UEV*

SVD gives you a representaiton of the data in a new basis. E is the singular value matrix, and then U and V* are right and left matrices. Low rank truncations of UEV* you end up with a good approximation of X with smaller operators. This is one of many types of change of bases like the fourier transform, laplace transform, and discrete cosine transform, that are useable for all sorts of problems. 

Compressed sensing sums it up as

    x = C psi s = theta s

Where C is a Gaussian or Bernouli random matrix, psi is your change of basis, and s is a sparse vector. Theta is just C and psi. Now Q and K are randomly initialized any way, and during the learning process Q,K,V reach an optimal set -- so it's closer to this stuff than one would imagine from a handwaving argument. 

For more information on this topic check out Steve Brunton's books. For this repo though, it's why I take a look at FFTs on the token sequences. Might be useful down the line. 

# Scaling
Then we start seeing all this research on hyperscaling. The 2019 Deep double descent paper found a regime where model test loss continues to improve with more parmeters, more data, and longer training [3]. Before that, the conventional wisdom from statistical learning approaches was that models would just over fit unless the data far exceeds the parameters; this is now an argument for intentional overparameterization. The 2020 Scaling Laws paper, further quantified this and showed that performance predictably scales logarithmically with data, parameters and compute [4]. Within this body of work, the message seems to be to scale until we can no longer get any kind of improvement.  

But in practice, virtual machines cost money, and your implementation leaves memory on the table. In 2022, we get FlashAttention which is an improvement on the memory implementation of attention blocks, and focuses on improving I/O [5]. Later in 2022 NVIDIA publishes on FP8 formats [6]. Up to this point we had seen a lot of floating point 32 and 64, but this paper keeps the model parameters in an 8 bit floating point type, and uses mixed types for the optimizer tracking and gradient updates. Despite the loss of precision with the parameters, they were able to achieve minimal errors in classification benchmarks. So Goldberg taught us, we need the precision for regression and signals, this new format says, well hold on, for classifiers we don't need as much. NVIDIA has a blog on this model quantization so its worth it to check it out and read more. [7]. 

Now the take away, is that the text models are giant next-token classifiers, so quantization seems to work well in language modeling as well. In practice, it's why you're seeing large companies release quantized parameter sets; it's cheaper to run those on the virtual machines for nearly the same performance than the full precision parameter set.

# Tokenizers
So, what are tokens? Why do they charge you by the 1 million tokens for API calls and what not? In text and NLP a token is a number that represents 3-5 characters on average. It's taken to mean the smallest, most semantically meaningful piece of text. So like syllables in some cases, word roots, whole words in somecases. At a byte level your computer recognizes 256 potential values. This covers lower and upper case letters, punctuation, control characters in ASCII and extended ASCII/Latin. 

So tokenizers are 2 part peices of code. They are 
1. Vocabulary learners. They find a set of text sequences that can be used as keys in a look up table when you want to go from text to numbers in a meaningful way.
2. An encoder/decoder pair where you use the look up table to convert between text and numbers freely.  

After the tokenizer converts text to numbers, then we use embeddings to find meaningful vector spaces --> change of bases from 1 number domain to another, and then train a transformer on top to go from embeddings to a new vector space or to a classifier. But since tokenizers can represent sequences of text as well as individual characters, they add a layer of compression to the text data. 

Two tokenizers that come up all over the literature are WordPiece and Byte Pair encoding. Both of these are bottom up algorithms that learn sequences from smaller sub words first. Both have greedy implementations. For word piece it picks the longest matches first, while BPE starts with the bigrams (2 letters) that occur most often and grows sequences from there. 

Bidirectional encoder representation tranformer (BERT) and versions of BERT all use WordPeice [10] BERT ued 800M words from BookCorpus, and then 2500M words from English wikipedia text. The draw back is that the probability distribution of commonly used words in the english language skews to shorter words; longer words tend to be used for domain specific text. It's why we have Term Frequency Inverse Document Frequence (TFIDF) for NLP and key word finding. So for example, Doestoevsky's books, and books on his writing are going to have different word and token distributions than books on and journal articles on malaria. And malaria data is going to change from the 1900s to today because of the introduction of novel medical technology used in its study and treatment. If the language and its meaning shift, so does to token set you need to describe that language. That's why we have custom BERT models like BioBert, which was trained on 4.5B words from PubMed, 13.5B words from PMC [11]. There's also PyschBert (but it's behind a pay wall) [12]. This group actually uses their own byte pair encoding because they notice that the available tokenizers were splitting medical terms into to many peices [13]. So there is good rationale to own everything from the data set, to the tokenizer, and even training the transformer in cases where you need custom text segmentation / classification / sentiment analysis. 

Now, As far as token counts go, that bioBertCorpus is 18 billion words (10^9). Say its like 2 tokens per word so 36 billion tokens. Then like 3 characters per token so like 108 billion characters -- that'd but us int he ball park of a bout 100 Gigabytes give or take for the whole set. Now, BPE is derived from a compression algorithm, so these seqeunces get shorter the more compressed the represetation is. That's useful though because EHRs are like 16k+ tokens a pop, with about 50% of that data being duplicated notes [19]. 

With all that in mind, the C++ implementation I did in the repo that uses a doubly linked list, a static counter, and a priority queue, ends up taking 5 seconds to train for 512 steps on 1.7 million characters (1.7MB). This is good enough for this proof of concept, but at like 1.47 billion words it'd take a lot longer. But you can wrap it in a python or bash script to to parallelize this thing and then just aggregate the token sets at the end. That being said, there is an optimal BPE tokenizer that makes use of a set of arrays to handle this, and it's much faster. so likely it's something good to implement in the future. 

* theres also work comparing BPE and word peice for tokenization in other languages like turkish [14]. so multilingual models, and applications where people work together in various languages are relavnet here. 

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

[19] Steinkamp J, Kantrowitz JJ, Airan-Javia S. Prevalence and Sources of Duplicate Information in the Electronic Medical Record. JAMA Netw Open. 2022;5(9):e2233348. doi:10.1001/jamanetworkopen.2022.33348