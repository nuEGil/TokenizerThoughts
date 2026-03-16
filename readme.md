Intro section in progress but this is the rough sketch. -- still need to make a step to ingest the learned token sets and use that in a python script to run the tokenization.... also need to try to compare token sets across different books... and then do a histogram of the sequence legnths.... in progress. 

# Abstract
Transformers and scaling seems to be everything. There's a systems level approach to the optimization probelm that involves the tokenizer step. so scale, but scale efficiently. --

# Contents
1. Floating point operation precision. 
2. Brunton's data driven science and engineering -> what SVD and FFT mean for text
3. Bigram tokenization and reconstruction errors
4. Does a better tokenizer help with this problem
5. Tokenizer optimization and the C++ rabbit hole
6. Token sets on individual docs vs an aggregated set. 

# Floating point operation and precision
* Goldberg published a paper on the IEEE floating point standard to provide better support for floating point [1]. It's 1991 and it's a while before AlexNet kicks off a DL renaissance. Computational modeling is more concerned with physics based simulations. You could think about solving differential equations involved in compressible fluids mechanics, and structural mechanics for things like designing novel aircraft wings or for better flight controls systems. It could be computational software to compute the shear forces on a bridge or a sky scraper -- things of this nature.
* 2017 we get the transformer achitecture in attention is all you need [2]. Offers more representational capacity than we have from previous models --> look at the math behind the attention block (after the dash is a rabit hole for that brunton section)- looks like SVD -- so a learnable kernel svd basically. SVD is just one of many bases, so this gets
* 2019 Deep double descent [3]-- hyper scaling starts as we see a regime where model testing loss gets way better than we thought - originally we thought the models would just overfit. this is apparently not the case. -- demonstrated for resnet and transformer - i think, need to check 
* 2020 Scaling Laws for Neural Language models [4]- we can predictably see that the models will get logarithmic scaling with more data, more parameters, and more training... so push that till we can't  -- seems like the mesage. 
* 2022 - Flash attention [5] ok we know that we are going to scale, but can we be more efficient about it?
* 2022 - FP8 formats [6] -- now we are back to gold berg, since we are concerned about classifiers we can actually get rid of a alot of the floating point precision in the parameters -- keep it int the optimizer and in the gradients, and get models that still train well. More on model quantization [7].
* Wikipedia on test functions for optimization problems - this gives some intuition and test cases for whatever your optimizer is and whether your loss function is learnable.[8]. Pick a function, we will see convexity, high local optima, and saddle points. Then things change as these functions get into higher dimenssions. Any loss function you write for a deep learning problem is gonna run into these types of issues in high dimensions.  
* AdamW - Adam was adaptive momentum, you can decouplpe the weight decay regularization...[9]

Two things to note here. 
1. Deep learning has always been about a numerical solution to an analyitic approximation problem. We know we can approximate any function because theres a proof for that, but now train it...
   
2. Assuming that the scaling thing is just a property of networks, then if we get more efficient sub-blocks and scale those, then we should be able to get better models down the line.... This got me thiking about the tokenizer -> and now at the end of this short rabbit hole -- ended up seeing that tokenizers really act to compress the input in a meaningful way before you ever talk about training an embedding --> plus , from a probabilty stand point you should get different token sets on different data sub batches than you would 

[1] David Goldberg. 1991. What every computer scientist should know about floating-point arithmetic. ACM Comput. Surv. 23, 1 (March 1991), 5–48. https://doi.org/10.1145/103162.103163.

[2] Ashish Vaswani et al. 2017, Attention is all you need

[3] Preetum Nakkiran et al. Deep Double Descent: Where Bigger Models and More Data Hurt., 2019, arXiv, (4 December 2019), https://arxiv.org/abs/1912.02292.

[4] Jared Kaplan et al. 2020, Scaling Laws for Neural Language Models. (23 Jan 2020) https://arxiv.org/abs/2001.08361

[5] Tri Dao et al. 2022, FlashAttention: Fast and Memory Efficient Exact Attention with IO-Awareness. arXiv, (27 May 2022), https://arxiv.org/abs/2205.14135.

[6] Paulius Micikevicius et al. 2022, FP8 Formats for Deep Learning. arXiv, (12 September 2022), https://arxiv.org/abs/2209.05433.

[7] https://developer.nvidia.com/blog/model-quantization-concepts-methods-and-why-it-matters/

[8] https://en.wikipedia.org/wiki/Test_functions_for_optimization

[9] Ilya Loshchilov et al. Decoupled Weight Decay regularization, arXiv, (14 Nov 2017) https://arxiv.org/abs/1711.05101 
