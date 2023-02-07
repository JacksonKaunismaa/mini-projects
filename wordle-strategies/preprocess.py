with open("./sedecordle_code.js", "r") as f:
    allowed, answers, _ = f.read().split("\n")

allowed_words = allowed[11:-1].split()
answer_words = answers[11:-1].split()
allowed_words = answer_words + allowed_words

with open("allowed_words.txt", "w") as f:
    f.write("\n".join(allowed_words))

with open("answer_words.txt", "w") as f:
    f.write("\n".join(answer_words))

