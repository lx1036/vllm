


# from prompt_toolkit import prompt
#
# if __name__ == '__main__':
#     answer = prompt('Give me some input: ')
#     print('You said: %s' % answer)



from InquirerPy import prompt

questions = [
    {"type": "input", "message": "What's your name:", "name": "name"},
    {
        "type": "list",
        "message": "What's your favourite programming language:",
        "choices": ["Go", "Python", "Rust", "JavaScript"],
    },
    {"type": "confirm", "message": "Confirm?"},
]
result = prompt(questions)
name = result["name"]
fav_lang = result[1]
confirm = result[2]
print(f"Hello {name}, you like {fav_lang} and {confirm}")





