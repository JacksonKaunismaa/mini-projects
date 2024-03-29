from collections import defaultdict
from collections.abc import Iterable
import itertools
from tqdm import tqdm

from . import permutation
from . import symbolic
from . import utils


class Group(set):  # includes both the elements of the Group and the rules of the representation
    def __init__(self, *elems, rules=None, generate=False, name=None, verbose=False, n=None):
        super().__init__(elems)
        self.singleton_rules = {}
        self.general_rules = defaultdict(list)
        self.symbols = set()
        self.simplify_cache = {}
        self.verbose = verbose
        self.name = name
        self.n = n
        self.quotient_map = None   # for quotient groups, how to map the elements to the simplest representative

        # if (elems and isinstance(elems[0], permutation.Permutation)):
        #     self.n = elems[0].group.n

        if rules:
            for rule in rules:
                pattern, result = rule.split("=")
                pattern_expr = self._parse(pattern, initial=True)
                result_expr = self._parse(result, initial=True)
                
                self._add_syms(pattern_expr, result_expr)  # so that we can generate the group later

                if len(pattern_expr) == 1 and  result_expr.is_identity:  # if symbol is cyclic, do this for efficiency
                    self.singleton_rules[pattern_expr[0].sym] = pattern_expr[0].exp
                    continue
                self.general_rules[len(pattern_expr)].append((pattern_expr, result_expr))    # map symbol -> (exponent, replacement)
        
        if generate is None:
            generate = not self.is_perm_group  # default is don't generate perm_groups, but generate everything else
        if generate:  # should probably only use this for finite groups
            self._generate_all()  # can't really handle infinite groups yet


    def _generate_all(self):  # generate all elements in the group
        if self.is_perm_group:
            if "symmetric" in self.name:
                self.update(permutation.Permutation(pt, self).simplify() for pt in itertools.permutations(list(range(self.n))))
            else:
                new_elems = [permutation.Permutation(pt, self).simplify() for pt in itertools.permutations(list(range(self.n)))]
                self.update([x for x in new_elems if x.parity == 0])
        else:
            elems = self.generate(*self.symbols)
            self |= elems
        
    def _same_group_type(self, other: "Group"):  # check if 2 Groups are subgroups of the same group
        if self.is_perm_group:
            if not other.is_perm_group:
                return False
            return self.n == other.n
        return self.general_rules == other.general_rules and self.singleton_rules == other.singleton_rules
        
    def _identity_term(self): # helper function to return an identity element, should only be called internally
        if self.is_perm_group:
            return permutation.Permutation([], self)
        else:
            return symbolic.Term(None, None)  # clients only use is_identity, and is_identity needs to be implemented everywhere
    
    def _identity_expr(self):  # helper function to return an expr, pretty much only used in generate
        if self.is_perm_group:
            return self._identity_term()  # for permutation groups, Terms and Expressions are kinda the same thing
        else:
            return symbolic.Expression([self._identity_term()], self)
    
    def _identity_group(self):   # helper function return a Group containing only an identity {Expression, Permutation}
        expr = self._identity_expr()
        return self.subgroup(expr)
    
    def _add_syms(self, *exprs):
        for expr in exprs:
            for term in expr:
                if not term.is_identity:
                    self.symbols.add(term.sym)

    def _parse(self, equation, initial=False):  # helper function for creating new expressions
        if self.is_perm_group:
            return permutation.Permutation(equation, self)
        else:
            return symbolic.Expression(equation, self, initial=initial)
    
    def subgroup(self, *elems):  # create an empty subgroup that has the same multiplication rules
        group = Group(*elems)
        set_these = ["singleton_rules", "general_rules", "n", "symbols", 
                     "verbose", "simplify_cache", "name", "quotient_map"]
        for var_name in set_these:
            if hasattr(self, var_name):
                obj = getattr(self, var_name)
                if var_name in ['quotient_map'] and obj:
                    obj = obj.copy()
                setattr(group, var_name, obj)
        return group
    
    def evaluate(self, equation):
        if isinstance(equation, str):
            return self._parse(equation).simplify()
        elif isinstance(equation, (list,tuple)):
            return [self.evaluate(s) for s in equation]  # => recursive
        else: # ie. Expression, Permutation, Term
            return equation

    def copy(self):
        return self.subgroup(*self)
    
    def __iter__(self, track=False):
        if not self.has_elems:
            print("Warning: you are trying to iterate over an empty group")
        iterator = super().__iter__()
        if track:
            return tqdm(iterator)
        return iterator
    

    # Properties


    @property
    def is_perm_group(self):
        return self.n is not None  # permutation groups have n defined, others don't
    
    @property
    def has_elems(self):
        return len(self) > 0
    
    def is_subgroup(self, verbose=True):
        if not self.has_elems:
            return False
        for elem1 in self:
            for elem2 in self:
                if elem1/elem2 not in self:
                    if verbose:
                        print(f"{elem1=}, {elem2=} generates {elem1/elem2} not in subgroup")
                    return False
        return True
    
    def is_normal(self, subgroup, verbose=False):            
        if not subgroup.is_subgroup(verbose=verbose):
            if verbose:
                print("not even a subgroup")
            return False
        for h in subgroup:
            for g in self:
                if g * h / g not in subgroup:
                    if verbose:
                        print(f"group_elem={g}, subgroup_elem={h} generates {g*h/g} not in subgroup")
                    return False
        return True


    # Operations


    def __mul__(self, other):  # ie Group * [Expression, Group, str, list[str]] (right cosets)
        if isinstance(other, (symbolic.Expression, permutation.Permutation)):
            new_elems = self.subgroup()
            for elem in self:
                new_elems.add(elem * other)
            return new_elems
        elif isinstance(other, Group) and self._same_group_type(other):
            new_elems = self.subgroup()
            for e1 in self:
                for e2 in other:
                    new_elems.add(e1 * e2)
            return new_elems
        elif isinstance(other, str):
            elem = self.evaluate(other)
            return self * elem
        elif isinstance(other, list) and isinstance(other[0], str):
            elems = self.generate(*other)
            return self * elems
        else:
            return NotImplemented
        
    
    def __rmul__(self, other): # ie. Expression * Group (left cosets)
        if isinstance(other, (symbolic.Expression, permutation.Permutation)):
            new_elems = self.subgroup()
            for elem in self:
                new_elems.add(other * elem)
            return new_elems
        elif isinstance(other, str):
            elem = self.evaluate(other)
            return elem * self
        elif isinstance(other, list) and isinstance(other[0], str):
            elems = self.generate(*other)
            return elems * self
        else:
            return NotImplemented
    

    def __truediv__(self, other): # ie. Group / {Term, Permutation}
        if isinstance(other, Group):
            if not self._same_group_type(other):
                raise ValueError("Incompatible group types {self.name} and {other.name}")
            if not self.is_normal(other):
                raise ValueError("Attempting to quotient by a non-normal subgroup")
            cosets = self.find_cosets(other)
            # print("cosets", cosets)
            quotient_map = {x: representative for representative, coset in cosets.items() for x in coset}
            reprs = cosets.keys()
            quotient = self.subgroup(*reprs)
            for representative in quotient_map.values():
                representative.group = quotient  # update the group in the map to the new, correct thing
            quotient.quotient_map = quotient_map
            return quotient
        
        elif isinstance(other, (symbolic.Expression, permutation.Permutation)):
            return self*other.inv()
        elif isinstance(other, list) and isinstance(other[0], str):
            elems = self.generate(*other)
            return self / elems
        else:
            return NotImplemented
        
    def __and__(self, other):  # make sure to cast to a Group object
        return self.subgroup(*super().__and__(other))
    
    def __or__(self, other):
        return self.subgroup(*super().__or__(other))
    

    def generate(self, *exprs) -> "Group":
        if len(exprs) == 0:
            return self._identity_group()
    
        flat_exprs = set()
        for expr in exprs:
            if isinstance(expr, str):
                flat_exprs.add(self.evaluate(expr))
            elif isinstance(expr, Group):
                flat_exprs |= expr
            else:
                raise ValueError(f"Unknown type '{type(expr)}' in exprs list")
            
        frontier = self.subgroup(*flat_exprs)
        visited = self.subgroup()
        # print("frontier", frontier)
        while len(frontier) > 0:  # BFS
            start = frontier.pop()
            # print("checking elem", start)
            for elem in flat_exprs:
                next = start*elem
                if next not in visited:
                    # print("found new node", next)
                    frontier.add(next)
                    visited.add(next)
                    #yield next  # so that we can do infinite groups as well
        return visited
    
        
    def centralizer(self, elems):
        elems = self.evaluate(elems)
        if not isinstance(elems, Iterable):
            elems = [elems]
        commuters = self.subgroup()
        for candidate in self:
            for pt in elems:
                if pt*candidate != candidate*pt:
                    break
            else:
                commuters.add(candidate)
        return commuters
    

    def center(self):
        return self.centralizer(self)
    

    def conjugacy_class(self, elem, paired=False, track=False):
        reachable = []
        generators = []  # the associated list of elements that generate each coset/element in "reachable"
        elem = self.evaluate(elem)
        for other in self.__iter__(track=track):
            new_elem = other * elem / other
            #print(other, "generates", new_elem)
            if new_elem not in reachable:  
                reachable.append(new_elem)
                generators.append(other)
            elif paired:  # if we want to know what to conjugate with to get each element in the conj_class, 
                idx = reachable.index(new_elem) # then set paired=True. This bit just picks the 'simplest' such element
                if utils.simpler_heuristic(other, generators[idx]):
                    generators[idx] = other
        if paired:
            return dict(zip(generators, reachable))
        elif not isinstance(elem, Group):
            return self.subgroup(*reachable)
        else:
            return reachable
    

    def orbit(self, base_elem):
        base_elem = self.evaluate(base_elem)

        reachable = self.subgroup()
        elem = base_elem
        reachable.add(elem)
        while not elem.is_identity:
            elem = elem*base_elem
            reachable.add(elem)
        return reachable


    def normalizer(self, elems): # no need to do .evaluate here, since we .generate anyway
        if not isinstance(elems, Group):
            elems = self.generate(elems)
        commuters = self.subgroup()
        for candidate in self:
            for elem in elems:
                if candidate*elem/candidate not in elems:
                    break
            else:
                commuters.add(candidate)
        return commuters
    
    def normal_closure(self, elems):  # return smallest normal subgroup that contains `elems`
        if not isinstance(elems, Iterable):
            elems = self.generate(elems)
        expanded = self.subgroup()
        for g in self:
            expanded |= g * elems / g
        # print(expanded, "expanded")
        return self.generate(expanded)
    
    def normal_core(self, elems):  # return largest normal subgroup contained in `elems`
        if not isinstance(elems, Iterable):
            elems = self.generate(elems)
        expanded = self.subgroup(*elems)
        for g in self:
            expanded &= g * elems / g
        return expanded
    

    def find_cosets(self, coset: "Group", left=True):
        cosets = {}
        full_group = self.copy()
        while len(full_group) > 0:
            elem = full_group.pop()
            if left:
                new_coset = elem * coset
            else:
                new_coset = coset * elem
            if new_coset not in cosets.values():
                best_representative = elem  # heuristically find the simplest representative
                for representative in new_coset:
                    if utils.simpler_heuristic(representative, best_representative):
                        best_representative = representative
                cosets[best_representative] = new_coset
                full_group = full_group - new_coset
        return cosets
    
    def commutator(self, track=False):
        elems = self.subgroup()
        for x in self.__iter__(track=track):
            for y in self:
                elems.add(x * y / x / y)
        return self.generate(elems)
