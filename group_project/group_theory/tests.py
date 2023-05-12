from .group_theory import *

def main():
    # Tests for permutation.py
    fours = get_all_permutations(4)
    a4 = generate([Permutation([[0,1,2]], n=4),
                Permutation([[1,2,3]], n=4),
                ])
    s4 = get_all_permutations(4)
    for perm in s4:
        print(perm, perm.cycle_type, perm.parity)
        if perm.parity == 0:
            assert perm in a4
    for perm in a4:
        assert perm.parity == 0
    print("##"*20)
    for perm in fours:
        print(perm, perm.cycle_type)
    print("##"*20)

    def main():
    gr = Group(["r8 = e",
                "f2 = e",
                "f r = r7 f",])

    gr.verbose = False

    # Tests for operations.py and groups.py
    e0 = "e e e e f e e e e e e e f e e e", "e"
    e1 = "r25 f1 r2 r4 r-17 f1 f2 t12", "r4 t12" 
    e2 = "e", "e"
    e3 = "1", "e"
    e4 = "f2 r8 e", "e"
    e5 = "r1", "r"
    e6 = "r2 f", "r2 f"
    e7 = "r3 f e r2 f e f", "r f"
    e8 = "r9 f r r r r", "r5 f"
    e9 = "r r r2 f r3 f f r5 r4 r", "r7 f"
    for v in [e0, e1, e2, e3, e4, e5, e6, e7, e8, e9]:
        parsed = gr.parse(v[0])
        answer = gr.parse(v[1])
        simplified = parsed.simplify()
        print(v, "-->", parsed, "-->", simplified)
        assert answer == simplified

    # Q8_group = Group(["i2=n",  # TODO: implement this using matrices over C somehow?
    #                  "j2=n",
    #                  "k2=n",
    #                  "n2=e",
    #                  "i j=k",
    #                  "i n=n i",
    #                  "j n=n j",
    #                  "k n=n k"])
    # Q8_group.verbose=True
    # print(Q8_group.parse("i j j j i k j i k j i k j i k j i k j i k j i k j i k i j k i j k j i k j i k j k").simplify())
    gr1 = default_groups("dicyclic", 8)
    r_expr = gr1.parse("r")
    s_expr = gr1.parse("s")
    gr1_elems = generate(r_expr, s_expr)
    print(gr1_elems, len(gr1_elems))
    r_orbit = generate(r_expr)
    print(is_subgroup(r_orbit))
    r_orbit.pop()
    print(is_normal(r_orbit, gr1_elems))


if __name__ == "__main__":
    main()




if __name__ == "__main__":
    main()