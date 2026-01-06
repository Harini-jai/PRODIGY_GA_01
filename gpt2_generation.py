from transformers import GPT2Tokenizer,GPT2LMHeadModel
tokenizer=GPT2Tokenizer.from_pretrained("gpt2")
model=GPT2LMHeadModel.from_pretrained("gpt2")
prompt="Once upon a time"
inputs=tokenizer.encode(prompt,return_tensors="pt")
outputs= model.generate(
    inputs,
    max_length=100,
    num_return_sequences=1,
    no_repeat_ngram_size=2,
)
print(tokenizer.decode(outputs[0],skip_special_tokens=True))
