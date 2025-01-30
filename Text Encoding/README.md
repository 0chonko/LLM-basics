# Tweak and Question
- Try tokenizing a multi-lingual text (e.g., German/English).
- How does BPE handle unknown characters?
- How does token reuse work with dynamic vocab?
- What are the limitations of this approach vs. BPE?
- How does performance scale with longer texts?
- Can you optimize the dynamic tokenizer’s speed?

- Describe the trade-offs between BPE and T-Free:
    BPE is efficient and widely used but has a fixed vocabulary. T-Free can adapt dynamically but may be slower and more complex.
- Why might T-Free be better suited for multilingual LLMs?:
    T-Free can handle new words and characters dynamically, making it more flexible for multilingual support.
- How could dynamic tokenization impact memory usage during inference?:
    Dynamic tokenization might use more memory due to the need to store and manage a dynamic vocabulary.    



