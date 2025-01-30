# This simplified tokenizer uses a hash function to dynamically assign token IDs 
# to words. It simulates the adaptive nature of T-Free, which can handle new words 
# without a fixed vocabulary

import hashlib  

class DynamicTokenizer:  
    def __init__(self, vocab_size=256):  
        self.vocab = {}  
        self.vocab_size = vocab_size  
        self.next_token_id = 0  

    def hash_token(self, token):  
        return int(hashlib.md5(token.encode()).hexdigest(), 16) % self.vocab_size  

    def encode(self, text):  
        tokens = []  
        for word in text.split():  
            if word not in self.vocab:  
                self.vocab[word] = self.next_token_id  
                self.next_token_id = (self.next_token_id + 1) % self.vocab_size  
            tokens.append(self.vocab[word])  
        return tokens  

# Example  
dynamic_tokenizer = DynamicTokenizer(vocab_size=64)  
texts = [  
    "Aleph Alpha's research",  
    "Semantic text encoding",  
    "Multilingual support"  
]  

for text in texts:  
    print(f"Text: {text}")  
    print(f"Tokens: {dynamic_tokenizer.encode(text)}\n")  