# ByteLevelBPETokenizer: This tokenizer is used to train a BPE model on custom text data. It splits the text into subwords based on frequency pairs.
# GPT2Tokenizer: This is a pre-trained tokenizer from Hugging Face’s transformers library, which uses BPE and is trained on a large corpus.

# Other types
    # Byte-level BPE, as used in GPT-2
    # WordPiece, as used in BERT
    # SentencePiece or Unigram, as used in several multilingual models

from tokenizers import ByteLevelBPETokenizer  
from transformers import GPT2Tokenizer  

# Initialize BPE tokenizer  
tokenizer = ByteLevelBPETokenizer()  
tokenizer.train(["Study of Aleph Alpha's T-Free method.", "This is a sample sentence for tokenization."])  

# Example text  
text = "Aleph Alpha's T-Free is revolutionary!"  
encoded = tokenizer.encode(text)  
print("Original text:", text)  
print("Encoded tokens:", encoded.tokens)  
print("Token IDs:", encoded.ids)  

# Compare with Hugging Face's GPT2 tokenizer  
hf_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")  
hf_encoded = hf_tokenizer.encode_plus(text, return_tensors="pt")  
print("\nHugging Face GPT2 Tokenizer:")  
print("Encoded tokens:", hf_tokenizer.convert_ids_to_tokens(hf_encoded['input_ids'][0]))  
print("Token IDs:", hf_encoded['input_ids'][0])  