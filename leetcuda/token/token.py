


import tiktoken
# tiktoken.get_encoding()
enc = tiktoken.get_encoding("o200k_base")
print(enc.encode("hello world"))
assert enc.decode(enc.encode("hello world")) == "hello world"
