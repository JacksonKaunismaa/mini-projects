from collections import Counter

with open("./answer_words.txt", "r") as f:
    words = [x.strip() for x in f.readlines()]
    word_counts = {w:Counter(w) for w in words}


while True:
    greens_inpt = input("Greens ('b1 r2' -> br___): ")
    if greens_inpt:
        greens = {int(x[1])-1: x[0] for x in greens_inpt.split(" ")}
    else:
        greens = {}


    # example here means the following:
    # there is at least 1 a and there are no a's in position 1 or 3
    # there is at least 2 d and there are no d's in position 1 or 4
    # there are exactly 2 c and there are no c's in position 2 or 5
    yellows_inpt = input("Yellows (include greens in bound 'a1:1,3 d2:1,4 2c:2,5'): ")
    if yellows_inpt:
        yellows = {}
        for item in yellows_inpt.split(" "):
            let,positions = item.split(":")
            positions = [int(n)-1 for n in positions.split(",")]
            if let[0].isdigit():  # exact
                yellows[let[1]] = ("exact", int(let[0]), positions)
            else: # lower bound
                yellows[let[0]] = ("lower", int(let[1]), positions)


    grays_inpt = input("Grays ('uhihxjirch'): ")
    if grays_inpt:
        grays = set(grays_inpt)
    else:
        grays = set()

    def criterion(w):
        if set(w) & grays or (not all(w[pos] == let for pos,let in greens.items())):  # if greens and grays satisfied
            return False

        counts = word_counts[w] # check yellows
        for let, (bound_type, bound, positions) in yellows.items():
            if bound_type == "lower":
                if counts[let] < bound:
                    return False
            elif bound_type == "exact":
                if counts[let] != bound:
                    return False
            if any(w[pos] == let for pos in positions):
                return False
        return True
    print("Filtering words...")
    remaining_words = list(filter(criterion, words))
    print(remaining_words[:min(len(remaining_words), 30)])
