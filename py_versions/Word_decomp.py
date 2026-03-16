import os
import re
import numpy as np
import matplotlib.pyplot as plt

def normalize_dashes(text):
    # Covers hyphen, en dash, em dash, figure dash, horizontal bar, etc.
    dash_pattern = r'[\u002D\u2010-\u2015\u2212\uFE58\uFE63\uFF0D]'
    return re.sub(dash_pattern, '-', text)

def normalize_quotes(text):
    # Convert curly quotes, guillemets, and other Unicode doubles to plain "
    quote_pattern = r'[\u201C\u201D\u201E\u201F\u2033\u2036\u00AB\u00BB]'
    return re.sub(quote_pattern, '"', text)

def normalize_whitespace(text):
    # Matches tabs, newlines, and carriage returns (one or more)
    # Replaces the entire cluster with 4 spaces
    return re.sub(r'[\t\n\r]+', '    ', text)

class Standardizer():
    def __init__ (self, x):
        self.max = np.max(x)
        # x = x/self.max
        self.sig  = x
        self.mean = np.mean(x)
        self.std  = np.std(x)
        
    def __repr__(self):
        return f"mean: {self.mean}\nstd: {self.std}\n"
    
    def exe(self, x = None):
        if x is not None:
            return (x - self.mean) / self.std
        else:
            return (self.sig - self.mean) / self.std
    
    def inv_exe(self, x = None):
        if x is not None:
            return (x * self.std) + self.mean
        else:
            return (self.sig * self.std) + self.mean
        
class Normalizer():
    def __init__ (self, x):
        self.max = np.max(x)
        self.min = np.min(x)
        self.sig  = x
        
    def __repr__(self):
        return f"max: {self.max}\nmin: {self.min}\n"
    
    def exe(self, x = None):
        if x is not None:
            return (x - self.min) / self.max
        else:
            return (self.sig - self.min) / self.max
    
    def inv_exe(self, x = None):
        if x is not None:
            return (x * self.max) + self.min
        else:
            return (self.sig * self.max) + self.min

def plotting(sig, freqs, tag =''):
    plt.figure(figsize=(10,5))
    plt.title(f'mean_flag={tag}')
    plt.subplot(3,1,1)
    plt.plot(sig)
    plt.subplot(3,1,2)
    plt.plot(np.real(freqs)) ## care ful 
    plt.xlabel('real value amp')
    plt.subplot(3,1,3)
    plt.plot(np.angle(freqs))
    plt.savefig(f'imgs/signals_{tag}.png')

def GetText(x):
    with open(x, "r", encoding='utf-8') as f:
        txt = f.read()
        
        # This one line now handles tabs, \n, and \r
        txt = normalize_whitespace(txt)
        txt = normalize_dashes(txt)
        txt = normalize_quotes(txt)
        return txt

class Tokenize_Obj():
    # probably only want one of these. 
    def __init__(self, gram_tag = 'bigram'):
        self.gram_tag = gram_tag 
        # Define the character set that will feed the vocabulary. 
        charset = [chr(i)for i in range(32, 126, 1)] 
        self.charset = charset 

        # Switch for choosing between bigrams and trigrams    
        if gram_tag == 'trigram':
            # print('setting up trigram')
            gram_set = lambda x : [x0+x1+x2 for x0 in x for x1 in x for x2 in x]
        else:
            # print('defaulting to bigram')
            gram_set = lambda x : [x0+x1 for x0 in x for x1 in x]

        # Define ngrams
        ngram = sorted(gram_set(charset))
        self.char_len = len(ngram[0])

        self.vocabulary_size = len(ngram) # needed 

        # dictionary to convert between characters and indices 
        self.string_to_int = {ch : i for i, ch in enumerate(ngram)}
        self.int_to_string = {i : ch for i, ch in enumerate(ngram)}

    def __repr__(self):
        return f"n_gram: {self.gram_tag}\ncharset: {''.join(self.charset)}\nvocabulary_size: {self.vocabulary_size}\n"
    
    # make encoder and decoder functions
    def encode(self, s):
        tokens = []
        for si in range(0, len(s), self.char_len):
            substr = s[si:si+self.char_len]   
            # handle out of index tokens. we dont ever use _ so thats the blank       
            tokens.append(self.string_to_int.get(substr, self.string_to_int['.'*self.char_len]))
        return tokens

    def decode(self, x):
        # 1. Round to nearest integer (99.9 -> 100)
        # 2. Convert to int for dictionary lookup
        # 3. Use .get() to handle outliers
        return ''.join(self.int_to_string.get(int(round(float(i))), '?' * self.char_len) for i in x)
        
class badTokenize_Obj(Tokenize_Obj):
    def decode(self, x):
        return ''.join(self.int_to_string.get(int(i), '?' * self.char_len) for i in x)

def task1(TokO, tok_tag = ''):
    # 1. set up the tokenizer
    
    #2. parse the text and tokenize it  
    book_dir = '/mnt/g/data/ebooks_public_domain/books'
    ff = os.path.join(book_dir, 'Crime_and_Punishment_.txt')
    text_ = GetText(ff)
    tokens = TokO.encode(text_)
    new_text = TokO.decode(tokens)

    win = 256
    start_ = 0
    stop_ = start_+win
    print('==========================')
    print('Encoding and decoding text')
    print('==========================')
    print('\nOriginal text ')
    print(text_[start_: stop_])

    print('\nReconstruction ')    
    print(new_text[start_: stop_])
    
    
    tokens = np.array(tokens, dtype= np.float64)
    # print('\n100 tokens from bigram encoding')
    # print('token set shape ', tokens.shape)
    # print(tokens[start_: start_+100])

    subtask_base(tokens, win, TokO, tag = 'base'+tok_tag)
    subtask_base_abs(tokens, win, TokO, tag = 'base_abs'+tok_tag)
    subtask_hard_lpfilt(tokens, win, TokO, tag = 'hlpfilt'+tok_tag)
     
    # Now can we use the FFT to do any thing? 
    # main question is can we even perfectly reconstruct the token set?

def subtask_base(tokens, win, TokO, tag = 'base'):
    stoks = tokens

    spectrogram = []
    for i in range(0,tokens.shape[0] - win, win):
        ff = (1./win)*np.fft.fft(stoks[i:i+win])
        spectrogram.append(ff)
    spectrogram = np.array(spectrogram)
    print(spectrogram.shape)
    plotting(stoks[0:win], spectrogram[0], tag)
    
    new_text= []
    for i in range(spectrogram.shape[0]):
        # have to use real not abs. abs will introduce errors 
        tt = (1.*win*np.fft.ifft(spectrogram[i,:])).real
        new_text.append(tt)
    
    new_text = np.array(new_text)
    scramble = TokO.decode(new_text[0])
    
    print('==========================')
    print("Result of FFT followed by iFFT - real")    
    print('==========================')
    print(scramble)

def subtask_base_abs(tokens, win, TokO, tag = 'base_abs'):
    stoks = tokens

    spectrogram = []
    for i in range(0,tokens.shape[0] - win, win):
        ff = (1./win)*np.fft.fft(stoks[i:i+win])
        spectrogram.append(ff)
    spectrogram = np.array(spectrogram)
    print(spectrogram.shape)
    plotting(stoks[0:win], spectrogram[0], tag)
    
    new_text= []
    for i in range(spectrogram.shape[0]):
        # have to use real not abs. abs will introduce errors 
        tt = np.abs(1.*win*np.fft.ifft(spectrogram[i,:]))
        new_text.append(tt)
    
    new_text = np.array(new_text)
    scramble = TokO.decode(new_text[0])
    
    print('==========================')
    print("Result of FFT followed by iFFT - abs")    
    print('==========================')
    print(scramble)

def subtask_meansub(tokens, win, TokO, tag = 'msub'):
    mean_token = np.mean(tokens, dtype = np.float64)
    stoks = tokens - mean_token
    
    spectrogram = []
    for i in range(0,tokens.shape[0] - win, win):
        ff = (1./win)*np.fft.fft(stoks[i:i+win])
        spectrogram.append(ff)
    spectrogram = np.array(spectrogram)
    print(spectrogram.shape)
    plotting(stoks[0:win], spectrogram[0], tag)
    
    new_text= []
    for i in range(spectrogram.shape[0]):
        # have to use real not abs. abs will introduce errors 
        tt = (1.*win*np.fft.ifft(spectrogram[i,:])).real
        tt = tt + mean_token
        new_text.append(tt)
    
    new_text = np.array(new_text)
    scramble = TokO.decode(new_text[0])
    
    print('==========================')
    print("Result of FFT followed by iFFT -- mean sub")
    print('==========================')
    print(scramble)

def subtask_hard_lpfilt(tokens, win, TokO, tag = 'hlpfilt'):
    mean_token = np.mean(tokens, dtype = np.float64)
    stoks = tokens - mean_token
    
    spectrogram = []
    # zero out even a little 5% window
    st_ = int(0.975*win //2)
    en_ = int(1.025*win //2)
    for i in range(0,tokens.shape[0] - win, win):
        ff = (1./win)*np.fft.fft(stoks[i:i+win])
        ff[st_:en_] = 0.99*ff[st_:en_]
        spectrogram.append(ff)
    spectrogram = np.array(spectrogram)
    print(spectrogram.shape)
    plotting(stoks[0:win], spectrogram[0], tag)
    
    new_text= []
    for i in range(spectrogram.shape[0]):
        # have to use real not abs. abs will introduce errors 
        tt = (1.*win*np.fft.ifft(spectrogram[i,:])).real
        tt = tt + mean_token
        new_text.append(tt)
    
    new_text = np.array(new_text)
    scramble = TokO.decode(new_text[0])
    
    print('==========================')
    print("Result of FFT followed by iFFT -- h_lpfilt")
    print('==========================')
    print(scramble)

def subtask_hard_hpfilt(tokens, win, TokO, tag = 'hhpfilt'):
    mean_token = np.mean(tokens, dtype = np.float64)
    stoks = tokens - mean_token
    
    spectrogram = []
    # zero out even a little 5% window
    st_ = int(0.05*win)
    
    for i in range(0,tokens.shape[0] - win, win):
        ff = (1./win)*np.fft.fft(stoks[i:i+win])
        ff[st_:-st_] = 0.*ff[st_:-st_]
        spectrogram.append(ff)
    spectrogram = np.array(spectrogram)
    print(spectrogram.shape)
    plotting(stoks[0:win], spectrogram[0], tag)
    
    new_text= []
    for i in range(spectrogram.shape[0]):
        # have to use real not abs. abs will introduce errors 
        tt = (1.*win*np.fft.ifft(spectrogram[i,:])).real
        tt = tt + mean_token
        new_text.append(tt)
    
    new_text = np.array(new_text)
    scramble = TokO.decode(new_text[0])
    
    print('==========================')
    print("Result of FFT followed by iFFT -- h_hpfilt")
    print('==========================')
    print(scramble)

if __name__ == '__main__':
    TokO = Tokenize_Obj(gram_tag='bigram')
    print('==================')
    print('tokenizer obj info')
    print('==================')
    print(TokO)
    task1(TokO, tok_tag='_good')
    
    print('\n\n=================')
    print('Now with a bad decoder')
    print('=================')

    badTokO = badTokenize_Obj(gram_tag='bigram')
    task1(badTokO, tok_tag ='_bad')

    