from collections import deque
import itertools
import re
from tqdm import tqdm

from . import permutation
from . import groups
from . import symbolic


def sliding_window(iterable, n):  # standard itertools recipe
    it = iter(iterable)
    window = deque(itertools.islice(it, n), maxlen=n)  # maxlen => the first element of the window
    if len(window) == n:  # will be removed when we do the append operation
        yield tuple(window)
    for x in it:
        window.append(x)
        yield tuple(window)


# returns True if term1 is heuristically "simpler" than term2
def simpler_heuristic(term1, term2):
    if term1.is_identity or term2.is_identity:  # identity is simplest, do this to avoid NoneType issues
        return term1.is_identity

    if isinstance(term1, symbolic.Expression):
        if len(term1) < len(term2):  # shorter Expressions are better
            return True
        if sum(x.exp for x in term1) < sum(x.exp for x in term2): # smaller exponents are better
            return True
    elif isinstance(term1, symbolic.Term):
        if term1.exp < term2.exp: # smaller exponents are better
            return True
    elif isinstance(term1, permutation.Permutation):
        sum_cycle1, sum_cycle2 = sum(term1.cycle_type), sum(term2.cycle_type)
        if sum_cycle1 < sum_cycle2:  # fewer terms are preferred
            return True
        elif sum_cycle1 == sum_cycle2: # shorter terms are preferred
            total = sum(i*(c1-c2) for i, (c1, c2) in enumerate(zip(term1.cycle_type, term2.cycle_type)))
            if total < 0:  # total < 0 => c2 has its cycles "later" ie. have more elems in the cycle
                return True
        if str(term1) < str(term2):  # prefer terms that are alphabetically in order
            return True  # ie. (1 2 3 4) should be preferred over (1 3 2 4)
    return False


def factorize(n):  # from https://codereview.stackexchange.com/questions/121862/fast-number-factorization-in-python
    # some common primes, should mean we can handle reasonable cases very slightly faster
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    for fact in itertools.chain(primes, itertools.count(primes[-1],2)):
        if n == 1:
            break
        while n % fact == 0:
            n //= fact
            yield fact
        if fact**2 > n:  # this means n must be prime
            yield n
            break


def get_interesting_sizes(n):
    # return a list of exponents that would generate a different subgroup from a single element of order n
    # this only really works for single elements
    # factors = list(Counter(factorize(n)).items())  # so that order is fixed
    # yield 0  # give the option to not use this particular element
    # for exponents in  itertools.product(*(range(sz) for _,sz in factors)):
    #     elem = 1
    #     for (fact,_), amt in zip(factors, exponents):
    #         elem *= fact**amt
    #     yield elem

    return range(n)  # this is the actual safest


def find_subgroups(group):  # only works for finite groups
    if isinstance(group, groups.Group):
        generators = group.singleton_rules  # generators should be a dict of symbol: orbit_of_symbol
    else:
        generators = {permutation.Permutation([[a, b]], n=group.n): 2  # generate group via adjacent transpositions 
                                    for a,b in zip(range(0,group.n-1), range(1,group.n))}
    generators = list(generators.items())  # so that order is fixed
    subgroups = {}
    # iterate over all possible exponents for all possible generators
    generator_iter = itertools.product(*(get_interesting_sizes(order) for _,order in generators))
    # generate each subgroup using the same number of generators as there are in the full group
    # if the same generator is repeated, that is equivalent to only using 1 generator
    # if generators are specified in a different order, they are equivalent to each other as well
    # thus, it makes the most sense to consider subgroup generators as a set() object, but we want them to be
    # hashable as well, so make it a frozenset
    for generator_terms in tqdm(itertools.product(generator_iter, repeat=len(generators))): 
        # inner comprehension is multiplying together generators of the full group to get one of the generators for the subgroup
        subgroup_generators = frozenset(group.parse(' '.join(f"{elem}{amt}" for (elem,_),amt in zip(generators, generator_term)))
                                        for generator_term in generator_terms)
        # outer comprehension is repeating that process for all the subgroup generators

        if subgroup_generators not in subgroups: 
            new_subgroup = group.generate(*subgroup_generators)
            #print(amts, "<" + ", ".join(str(x) for x in subgroup_generators) + ">", "=", new_subgroup)
            if new_subgroup and new_subgroup not in subgroups.values():  # only add unique subgroups
                subgroups[subgroup_generators] = new_subgroup
    return subgroups

def _default_groups(group_name, n, generate): # some definitions for common finite groups
    if group_name == "quaternion":
        group_name = "dicyclic"
        # n //= 2  # use the group explorer and video notation (different from notation in the slides)
    # symbolic groups definition
    rules_d =  dict(cyclic=[f"r{n} = e"],
                    dihedral=[f"r{n} = e",
                              f"f2 = e",
                              f"f r = r{n-1} f"],
                    dicyclic=[f"r{n} = e",
                              f"s4 = e",
                              f"s2 = r{n//2}",
                              f"s r = r{n-1} s"],
                    semi_dihedral=[f"r{n} = e",
                                   f"s2 = e",
                                   f"s r = r{n//2-1} s"],
                    semi_abelian=[f"r{n} = e",
                                  f"s2 = e",
                                  f"s r = r{n//2+1} s"],
                    abelian=[f"r{n} = e",
                             f"s2 = e",
                             f"s r = r s"],
                    )

    if group_name in ["symmetric", "alternating"]:  # use different settings for permutation groups since they are quite large
        return groups.Group(n=n, generate=generate, name=f"{group_name} {n}")  # and also have no symbolic rules
    else:
        if group_name in ["dicyclic", "semi_dihedral", "semi_abelian"] and n % 2 != 0:
            raise ValueError(f"n must be divisible by 2 for {group_name} group, it was {n} instead")
        return groups.Group(rules=rules_d[group_name], generate=generate, name=f"{group_name} {n}")


def get_group(query, generate=None):  # would use this in a "interactive" session, but probably useless
    extracted = re.search(r"([a-zA-Z]+) ?(\d+)", query.lower().strip())
    mappings = [(["d", "dihedral", "dih", "di"], "dihedral"),
                (["c", "z", "cyclic", "cyc"], "cyclic"),
                (["dic", "dicyclic"], "dicyclic"),
                (["semi-dihedral", "sd", "semi_dihedral"], "semi_dihedral"),
                (["semi-abelian", "sa", "semi_abelian"], "semi_abelian"),
                (["abelian", "ab"], "abelian"),
                (["quaternion", "quat", "q"], "quaternion"),
                (["perm", "permutation", "sym", "symmetric", "s"], "symmetric"),
                (["alt", "alternating", "a"], "alternating")]
    group_name = extracted.group(1).strip()
    for (alt_names, name) in mappings:
        if group_name in alt_names:
            group_name = name
            break
    else:
        print(f"Group name {group_name} not recognized")
        return
    size = int(extracted.group(2))
    return _default_groups(group_name, size, generate)

def quicktest():  # some quick and dirty tests
    # Test parsing and simplification a bit
    d8_alt = groups.Group(rules=["r8 = e",
                                "f2 = e",
                                "r f r = f",
                                "f r = r7 f"])
    d8 = get_group("d8")

    test_elems = [("e e e e f e e e e e e e f e e e", "e"),
                  ("r25 f1 r2 r4 r-17 f1 f2 f12", "r4"),
                  ("e", "e"),
                  ("1", "e"),
                  ("f2 r8 e", "e"),
                  ("r1", "r"),
                  ("r2 f", "r2 f"),
                  ("r3 f e r2 f e f", "r f"),
                  ("r9 f r r r r", "r5 f"),
                  ("r r r2 f r3 f f r5 r4 r", "r7 f")]
    for v in test_elems:
        parsed_alt = d8_alt._parse(v[0])
        parsed = d8._parse(v[0])
        assert parsed == parsed_alt

        answer_alt = d8_alt._parse(v[1])
        answer = d8._parse(v[1])

        evald = d8_alt.evaluate(v[0])
        evald_alt = d8.evaluate(v[0])
        
        simplified_alt = parsed_alt.simplify()
        simplified = parsed.simplify()

        assert answer == simplified == answer_alt == simplified_alt == evald == evald_alt
        assert str(answer) == str(simplified) == str(answer_alt) == str(simplified_alt) == str(evald) == str(evald_alt) == str(v[1])

    # Test multiplication (generated code)
    gr = get_group('dic 28')
    t1,t2,ans = gr.evaluate(('r5 s', 'r16', 'r17 s')); assert t1*t2 == ans
    t1,t2,ans = gr.evaluate(('r6', 'r26 s', 'r4 s')); assert t1*t2 == ans

    gr = get_group('sa 32')
    t1,t2,ans = gr.evaluate(('r29 s', 'r8 s', 'r5')); assert t1*t2 == ans
    t1,t2,ans = gr.evaluate(('r9 s', 'r31', 'r24 s')); assert t1*t2 == ans

    gr = get_group('sd 64')
    t1,t2,ans = gr.evaluate(('r33 s', 'r44', 'r53 s')); assert t1*t2 == ans
    t1,t2,ans = gr.evaluate(('r51 s', 'r2 s', 'r49')); assert t1*t2 == ans

    gr = get_group('dih 31')
    t1,t2,ans = gr.evaluate(('r20 f', 'r11 f', 'r9')); assert t1*t2 == ans
    t1,t2,ans = gr.evaluate(('r8 f', 'r7', 'r f')); assert t1*t2 == ans

    gr = get_group('cyc 29')
    t1,t2,ans = gr.evaluate(('r12', 'r3', 'r15')); assert t1*t2 == ans
    t1,t2,ans = gr.evaluate(('r20', 'r2', 'r22')); assert t1*t2 == ans

    gr = get_group('abelian 14')
    t1,t2,ans = gr.evaluate(('r8', 'r7 s', 'r s')); assert t1*t2 == ans
    t1,t2,ans = gr.evaluate(('r13 s', 'r4 s', 'r3')); assert t1*t2 == ans

    gr = get_group('quat 32')
    t1,t2,ans = gr.evaluate(('r10 s', 'r8', 'r2 s')); assert t1*t2 == ans
    t1,t2,ans = gr.evaluate(('r15 s', 'r7 s', 'r24')); assert t1*t2 == ans

    gr = get_group('sym 8')
    t1,t2,ans = gr.evaluate(('(1 4 7)(2 6 8 5 3)', '(1 7 2)(3)(4 5)(6 8)', '(1 5 3)(2 8 4)')); assert t1*t2 == ans
    t1,t2,ans = gr.evaluate(('(1 8 2 3 4)(5 7)(6)()', '(1 5 2 6)(3 8 4 7)', '(1 4 5 3 7 2 8 6)')); assert t1*t2 == ans

    gr = get_group('alt 8')
    t1,t2,ans = gr.evaluate(('(1 2)(3 7 8 4 6 5)', '(1 7 8 6 5 4 3 2)', '(2 7 6 4 5)(3 8)')); assert t1*t2 == ans
    t1,t2,ans = gr.evaluate(('(1 7 6 3 4 2)(5)(8)()', '(1 3)(2)(4 6 5 8)(7)()', '(1 7 5 8 4 2 3 6)')); assert t1*t2 == ans
    print("passed all tests")



# Test code generators

def rand_expr(group: groups.Group):
    import random
    if group.is_perm_group:
        res = list(range(group.n))
        random.shuffle(res)
        return permutation.Permutation(res, group)
    else:
        expr_str = ""
        for sym in group.symbols:
            expr_str += f"{sym}{random.randint(0,100)} "
        return group.evaluate(expr_str)


def rand_problem(group: groups.Group):
    t1 = rand_expr(group)
    t2 = rand_expr(group)
    return (str(t1), str(t2), str(t1*t2))


def multiplication_problem_template(group_name, problems):
    problem_text = [f"t1,t2,ans = gr.evaluate({problem}); assert t1*t2 == ans" for problem in problems]
    return "\n".join([f"gr = get_group('{group_name}')"] + problem_text)


def generate_problems():  # code generator for tests
    group_names = ["dic 28", "sa 32", "sd 64", "dih 31", "cyc 29", "abelian 14", "quat 32", "sym 8", "alt 8"]
    for group_name in group_names:
        print(group_name)
        get_group(group_name)
    group_selection = list(map(get_group, group_names))
    num_problems = 2
    probs = [[rand_problem(gr) for _ in range(num_problems)] for gr in group_selection]
    print("# Test multiplication (generated code)")
    for group_name,prob in zip(group_names, probs):
        print(multiplication_problem_template(group_name, prob))
        print()


