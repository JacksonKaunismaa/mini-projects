from collections.abc import Iterable
from . import permutation
from . import groups

# returns True if term1 is heuristically "simpler" than term2
def simpler_heuristic(term1, term2):
    if term1.is_identity or term2.is_identity:  # identity is simplest, do this to avoid NoneType issues
        return term1.is_identity

    if isinstance(term1, groups.Expression):
        if len(term1) < len(term2):  # shorter Expressions are better
            return True
        if sum(x.exp for x in term1) < sum(x.exp for x in term2): # smaller exponents are better
            return True
    elif isinstance(term1, groups.Term):
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
    reachable = groups.Group()
    elem = base_elem
    reachable.add(elem)
    while not elem.is_identity:
        elem = elem*base_elem
        reachable.add(elem)
    return reachable


def generate(*elems) -> groups.Group:
    if isinstance(elems[0], list):
        elems = elems[0]
    frontier = groups.Group(*elems)  # should retain the rules?
    visited = groups.Group()
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
    commuters = groups.Group()
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
    commuters = groups.Group()
    for candidate in all_elems:
        for elem in elems:
            if candidate*elem/candidate not in elems:
                break
        else:
            commuters.add(candidate)
    return commuters


def find_cosets(coset: groups.Group, full_group: groups.Group, left=True) -> list[groups.Group]:
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


def is_subgroup(elems: groups.Group):
    for elem1 in elems:
        for elem2 in elems:
            if elem1/elem2 not in elems:
                return False
    return True

def is_normal(elems: groups.Group, all_elems: groups.Group):
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
    return groups.Group(rules=rules_d[group_name], generate=True)


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