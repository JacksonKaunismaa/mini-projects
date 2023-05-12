from collections.abc import Iterable
from permutation import *
from groups import *

# returns True if term1 is heuristically "simpler" than term2
def simpler_heuristic(term1, term2):
    if term1.is_identity:  # identity is simplest
        return True
    
    if isinstance(term1, Expression):
        if len(term1) < len(term2):  # shorter Expressions are better
            return True
        if sum(x.exp for x in term1) < sum(x.exp for x in term2): # smaller exponents are better
            return True
    elif isinstance(term1, Term):
        if term1.exp < term2.exp: # smaller exponents are better
            return True
    elif isinstance(term1, Permutation):
        sum_cycle1, sum_cycle2 = sum(term1.cycle_type), sum(term2.cycle_type)
        if sum_cycle1 < sum_cycle2:  # fewer terms are preferred
            return True
        elif sum_cycle1 == sum_cycle2: # shorter terms are preferred
            total = sum(i*(c1-c2) for i, (c1, c2) in enumerate(zip(term1.cycle_type, term2.cycle_type)))
            if total < 0:  # total < 0 => c2 has its cycles "later" ie. have more elems in the cycle
                return True
    return False


def conjugacy_class(elem, all_elems):
    reachable = []
    generators = []  # the associated list of elements that generate each coset/element in "reachable"
    for other in all_elems:
        new_elem = other * elem / other
        #print(other, "generates", new_elem)
        if new_elem not in reachable:
            reachable.append(new_elem)
            generators.append(other)
        else:
            idx = reachable.index(new_elem)
            if simpler_heuristic(other, generators[idx]):
                generators[idx] = other
    return dict(zip(generators, reachable))


def orbit(base_elem):
    reachable = Coset()
    elem = base_elem
    reachable.add(elem)
    while not elem.is_identity:
        elem = elem*base_elem
        reachable.add(elem)
    return reachable


def generate(*elems) -> Coset:
    if isinstance(elems[0], list):
        elems = elems[0]
    frontier = Coset(*elems)
    visited = Coset()
    while len(frontier) > 0:  # BFS
        start = frontier.pop()
        #print("checking elem", start)
        for elem in elems:
            next = start*elem
            if next not in visited:
                #print("found new node", next)
                frontier.add(next)
                visited.add(next)
                #yield next  # so that we can do infinite groups as well
    return visited

def centralizer(elems, all_elems):
    if not isinstance(elems, Iterable):
        elems = [elems]
    commuters = Coset()
    for candidate in all_elems:
        for pt in elems:
            if pt*candidate != candidate*pt:
                break
        else:
            #print(commuters)
            commuters.add(candidate)
    return commuters


def normalizer(elems, all_elems):
    if not isinstance(elems, Iterable):
        elems = generate(elems)
    #all_perms = #get_all_permutations(max(elems, key=lambda x: x.n).n)
    commuters = Coset()
    for candidate in all_elems:
        for elem in elems:
            if candidate*elem/candidate not in elems:
                break
        else:
            commuters.add(candidate)
    return commuters


def find_cosets(coset: Coset, full_group: Coset, left=True) -> list[Coset]:
    arbitrary_elem = next(iter(coset))
    cosets = {arbitrary_elem.identity():  coset}
    full_group = full_group - coset
    while len(full_group) > 0:
        elem = full_group.pop()
        new_coset = elem * coset
        if new_coset not in cosets.values():
            best_representative = elem  # heuristically find the simplest representative
            for representative in new_coset:
                if simpler_heuristic(representative, best_representative):
                    best_representative = representative
            cosets[best_representative] = new_coset
            full_group = full_group - new_coset
    return cosets


def is_subgroup(elems: Coset):
    for elem1 in elems:
        for elem2 in elems:
            if elem1/elem2 not in elems:
                return False
    return True

def is_normal(elems: Coset, all_elems: Coset):
    if not is_subgroup(elems):
        return False
    for h in elems:
        for g in all_elems:
            if g * h / g not in elems:
                return False
    return True


def center(all_elems):
    return centralizer(all_elems)


def default_groups(group_name, n):      # some definitions for common finite groups
    rules_d =  dict(cyclic=[f"r{n} = e"],
                    dihedral=[f"r{n} = e",
                              f"f2 = e",
                              f"f r = r{n-1} f"],
                    dicyclic=[f"r{n} = e",
                              f"s4 = e",
                              f"r{n//2} = s2",
                              f"s r = r{n-1} s"],
                    semi_dihedral=[f"r{n} = e",
                                   f"s2 = e",
                                   f"s r = r{n//2-1} s"],
                    semi_abelian=[f"r{n} = e",
                                  f"s2 = e",
                                  f"s r = r{n//2+1} s"],
                    abelian=[f"r{n} = e",
                             f"s2 = e",
                             f"s r = r s"]
                    )
    if group_name in ["dicyclic", "semi_dihedral", "semi_abelian"] and n % 2 != 0:
        assert f"n must be divisible by 2 for {group_name} group, it was {n} instead"
    return Group(rules_d[group_name])


def get_group(query):  # would use this in a "interactive" session, but probably useless
    import regex as re  # since we use \p{L}

    extracted = re.search(r"(\p{L}+)(\d+)", query.lower().strip())
    mappings = [(["d", "dihedral", "dih", "di"], "dihedral"),
                (["c", "z", "cyclic", "cyc"], "cylic"),
                (["dic", "dicyclic"], "dicylic"),
                (["semi-dihedral", "sd", "semi_dihedral"], "semi_dihedral"),
                (["semi-abelian", "sa", "semi_abelian"], "semi_abelian"),
                (["abelian", "a", "ab"], "abelian")]
    group_name = extracted.group(1).strip()
    for (alt_names, name) in mappings:
        if group_name in alt_names:
            group_name = name
            break
    else:
        print(f"Group name {group_name} not recognized")
        return
    size = int(extracted.group(2))
    return default_groups(group_name, size)


def main():
    gr = Group(["r8 = e",
                "f2 = e",
                "f r = r7 f",])

    gr.verbose = False

    # some quick and dirty tests
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