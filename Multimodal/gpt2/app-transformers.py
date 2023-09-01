from transformers import pipeline, set_seed
generator = pipeline('text-generation', model='gpt2')
set_seed(42)
result = generator("Hello, My Name is Kimi.",
                   max_length=30, num_return_sequences=5)
print(result)

'''
[
    {'generated_text': 'Hello, My Name is Kimi.'}, 
    {'generated_text': 'Hello, My Name is Kimi.\n\nDear You\n\nYay! Good afternoon, I hope you enjoyed the game so much.\n'}, 
    {'generated_text': 'Hello, My Name is Kimi.\n\nYou know, if there was ever a time when we\'d need to call ourselves a "faux'}, 
    {'generated_text': 'Hello, My Name is Kimi.\n\nNoel: Welcome to the universe.\n\nKimi: Do you understand the concept?\n'}, 
    {'generated_text': 'Hello, My Name is Kimi. I live in Tokyo and live with my mum. We have been following the story of Nana, a young'}]

'''
