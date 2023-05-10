import itertools
from itertools import islice
from collections.abc import Iterable
from collections import defaultdict, deque

#def chunker(seq, size):  # general func for grouped
#    return (seq[pos: pos+size] for pos in range(0, len(seq), size))

def sliding_window(iterable, n):  # standard itertools recipe
    it = iter(iterable)
    window = deque(islice(it, n), maxlen=n)  # maxlen => the first element of the window
    if len(window) == n:  # will be removed when we do the append operation
        yield tuple(window)
    for x in it:
        window.append(x)
        yield tuple(window)


def tprint(obj, **kwargs):
    print(obj, type(obj), **kwargs)

IDENTITY_SYMBOLS = ["e", "1"]

class Term():  # a single instance of something like "r^3", r is the sym, 3 is the exp
    def __init__(self, sym, exp, group):
        self.sym = sym
        self.exp = exp
        self.group = group
        self.cyclic_rule = group.singleton_rules.get(sym, None)
        self.simplify()

    def simplify(self):
        if self.cyclic_rule:
            if self.exp < 0:
                self.exp += (self.cyclic_rule*(1+(-self.exp)//self.cyclic_rule))
            self.exp = self.exp % self.cyclic_rule
            if self.exp == 0:
                self.sym = None  # make self identity, not great since now we have 2 implementations of identity() (other is in Group)
        return self

    @property
    def is_identity(self):
        return self.sym is None

    def combine_like_terms(self, **kwargs):
        return self

    def __mul__(self, other):  # strictly for Term * Term
        if other.is_identity:
            return self
        if self.is_identity:
            return other

        if self.sym == other.sym:
            new_exponent = self.exp + other.exp
            return Term(self.sym, new_exponent, self.group)
        else:
            return Expression([self, other], self.group)

    def __repr__(self):
        if self.exp == 1:
            return self.sym
        if self.is_identity:
            return "e"
        return f"{self.sym}{self.exp}"

    def __eq__(self, other):
        return self.sym == other.sym and self.exp == other.exp

    def inv(self):
        return Term(self.sym, -self.exp, self.group)

    def __truediv__(self, other):
        return self * other.inv()

class Expression():
    def __init__(self, expr, group):
        self.group = group
        self.expr = expr
        if self.expr:
            if isinstance(self.expr[0], tuple): # turn rules into expressions
                self.expr = [Term(x[0], x[1], self.group) for x in self.expr] # handle identity?
        else:
            self.expr = [group.identity()]  # allowing empty expresions makes things buggy


    def windows_match(self, window, pattern):
        if (window[0].sym == pattern[0].sym and window[0].exp >= pattern[0].exp) and \
           (window[-1].sym == pattern[-1].sym and window[-1].exp >= pattern[-1].exp):
            for actual,expected in zip(window[1:-1], pattern[1:-1]):
                if actual != expected:
                    return False
            return True
        return False

    def simplify(self, max_iters=50):
        updated = True  # if we've applied a rule or not in the given iteration
        n = 0  # check total iterations so far
        while updated and n < max_iters:
            n += 1
            #print("curr expr is", self)
            updated = False
            for window_size in sorted(self.group.general_rules.keys(), reverse=True):  # check all possible window-sizes we have rules for
                if len(self) < window_size:
                    continue
                new_expr = Expression([], self.group)  # could be Expression([], self.group) as well

                window_iter = sliding_window(self.expr, window_size)  # current window we are looking to apply rules in
                last_posn = 0  # track where we've appended up to, so we can append missing ones at the end
                for window in window_iter:
                    #print("checking window of", window, type(window))
                    for pattern,result in self.group.general_rules[window_size]:  # check all possible rules at this given window
                        #print("\tchecking window against", pattern)
                        if self.windows_match(window, pattern):  # if the rule applies
                            #print("\t\twindow matches, proceeding with replacement...")
                            updated = True
                            #print("\t\tbefore replacing, new_expr is now", new_expr)
                            #print("\t\twill be adding", Expression([window[0]/pattern[0]] + result + [window[-1]/pattern[-1]], self.group))
                            # the result of applying the rule.
                            # filter identity to save a bit of compute (calculating translation doesn't filter for it
                            translation = Expression([window[0]/pattern[0]] + result + [window[-1]/pattern[-1]],
                                                     self.group)._filter_identity()
                            new_expr *= translation
                            #print("\t\tafter replacing, new_expr is now", new_expr)
                            #print("\t\twindow matched, advancing window by", window_size, "spaces")
                            try:  # skip window_size worth of windows because we've already used all those terms
                                last_posn += window_size
                                #print("\t\tnew last posn", last_posn)
                                window = [next(window_iter) for _ in range(window_size)][0]
                            except StopIteration:
                                #print("stop iter")
                                break # no need to check other rules if we've ran out of windows to look at
                            #print("\t\tafter advancing, new_expr is now", new_expr)
                            break
                    else:  # if we reach the end, we've checked all patterns and nothing worked, so just append 1 term and move the window
                        #print("\tbefore appending, new_expr is now", new_expr)
                        new_expr *= window[0]
                        last_posn += 1
                        #print("\tafter appending, new_expr is now", new_expr)

                #print("appending last window of", self[last_posn:])
                if last_posn != len(self): # append any missing terms that got skipped over because we moved window
                    new_expr *= self[last_posn:]
                #print("end_cycle new_expr is now", new_expr)
                self.expr = new_expr
            #combined = self.combine_like_terms()
            #if self != combined:
            #    updated = True
            #    self.expr = combined.expr
            #break
        return self.combine_like_terms()

    def combine_like_terms(self, n=None):
        curr_term = self.group.identity()
        #print("COMBINING", n)
        if n is None:
            new_expr = []
            for term in self.expr:
                curr_term = curr_term * term
                if isinstance(curr_term, Expression):
                    new_expr.append(curr_term[0])
                    curr_term = curr_term[1]
            new_expr.append(curr_term)
            return Expression(new_expr, self.group)
        else:   # if n set, then treat it as we have concatenated 2 expressions and we are working inwards out
            for i, (term1, term2) in enumerate(zip(self.expr[n-1::-1], self.expr[n:])):
                #print("looking at", term1, curr_term, term2)
                curr_term = term1 * curr_term * term2
                if isinstance(curr_term, Expression):
                    break
            else:
                curr_term = [curr_term]
            curr_term = self._filter_identity(curr_term)
            #print("Adding new terms, currently is", self)
            #tprint(self.expr[:n-i])
            #tprint(curr_term)
            #tprint(self.expr[n+i:], end="##########\n")
            #print(gr.parse("r25 f1 r2 r4 r-17 f1 f2 t12"))
            new_expr = self.expr[:n-i-1] + curr_term + self.expr[n+i+1:]
        self.expr = new_expr
        return self # Expression(new_expr, self.group)

    def _filter_identity(self, expr=None):  # remove all identity terms from an Expression or a list
        if expr is None:
            expr = self.expr
        if isinstance(expr, Expression):
            expr_terms = expr.expr
        else:
            expr_terms = expr
        return Expression(list(filter(lambda x: not x.is_identity, expr_terms)), self.group)

    def __getitem__(self, idx):
        return self.expr[idx]

    def __len__(self):
        return len(self.expr)

    def __radd__(self, other):  # for adding {list, Expression} + Expression
        if isinstance(other, list):
            return Expression(other + self.expr, self.group)
        else:
            return Expression(other.expr + self.expr, self.group)

    def __add__(self, other): # for adding Expression + {list, tuple, Expression}
        # ability to easily concatenate Expressions, as well as Expressions with lists
        if isinstance(other, list):
            return Expression(self.expr + other, self.group)
        elif isinstance(other, tuple):  # for adding windows when doing .simplify()
            return Expression(self.expr + list(other), self.group)
        else:
            return Expression(self.expr + other.expr, self.group)

    def __mul__(self, other): # self * other
        # other can be another Expression or a Term (__mul__ uses the left operand)
        if isinstance(other, Expression):
            other_expr = other.expr
        elif isinstance(other, tuple) or isinstance(other, list):
            other_expr = other
        elif isinstance(other, Term):
            other_expr = [other]
        new_expr = Expression(self.expr + other_expr, self.group)
        return new_expr.combine_like_terms(len(self))

    def __eq__(self, other):
        print(type(self), type(other))
        return all(self.expr == other.expr)

    def __repr__(self):
        return " ".join([str(t) for t in self.expr])


class Group():
    def __init__(self, rules):
        self.singleton_rules = {}
        self.general_rules = defaultdict(list)

        for rule in rules:
            pattern, result = rule.split("=")
            pattern_expr = self.parse(pattern)
            result_expr = self.parse(result)
            if isinstance(pattern_expr, Term) and result_expr.is_identity:
                # map symbol -> (exponent, replacement)  # make these Terms
                self.singleton_rules[pattern_expr.sym] = pattern_expr.exp
            else:
                if isinstance(result_expr, Term):
                    result_expr = Expression([result_expr], self)
                self.general_rules[len(pattern_expr)].append((pattern_expr, result_expr))

        #print(self.singleton_rules)
        #print(self.general_rules)

    def identity(self): # helper function to return an identity element
        return Term(None, None, self)

    def parse(self, equation):
        terms = equation.strip().split()
        start = self.identity()
        for t in terms:
            if t[0] in IDENTITY_SYMBOLS:
                next_term = self.identity()
            elif len(t) == 1:
                next_term = Term(t[0], 1, self)  # 1 is default exponent
            else:
                next_term = Term(t[0], int(t[1:]), self)
            #print("Currently is", (start, type(start)), "adding", next_term)
            start = start * next_term
        return start




        #return self.pretty(self.combine_terms(expr))
       #return self.combine_terms(expr)

class Permutation():
    def __init__(self, cycle_notation=None, result_notation=None, n=None):
        assert (cycle_notation is None) ^ (result_notation is None)
        notation = cycle_notation if cycle_notation is not None else result_notation

        #print(cycle_notation, result_notation, notation)

        self.n = n
        if cycle_notation is not None:
            if not n:
                self.n = 1+max(max(notation, key=max))  # technically a lower bound on n
            self.cycle = notation
        else:
            self.n = max(notation)+1
            shifted_notation = notation
            remain = set(range(self.n))
            curr_term = []
            self.cycle = []
            start_new = True  # [2, 5, 3, 1, 4]
            while remain: #[1,4,2,0,3])
                #print(f"{remain=}")
                if start_new:
                    start_new = False
                    elem = min(remain)
                    curr_term.append(elem)
                    remain.remove(elem)
                #print(elem, "goes to", end=" ")
                elem = shifted_notation.index(elem)
                #print(elem)
                if elem == curr_term[0]:
                    #print(f"cycle finished {curr_term=}")
                    self.cycle.append(curr_term.copy())
                    curr_term = []
                    start_new = True
                else:
                    curr_term.append(elem)
                    remain.remove(elem)
            self.cycle.append(curr_term.copy())
            #print(self.cycle)

        self.simplify()

    @property
    def is_identity(self):
        return not self.cycle

    @property
    def cycle_type(self):
        cycle_lens = [len(x) for x in self.cycle]
        num_cycles = []
        for i in range(self.n):
            num_cycles.append(cycle_lens.count(i+1))
        return num_cycles

    def __repr__(self):
        if not self.cycle:
            return "e"
        else:
            return "".join([f'({" ".join(map(lambda y: str(y+1), x))})'
                            for x in self.cycle])

    def simplify(self):
        #print(self.n, "n found")
        #print("cycle begins as", self.cycle)
        remain = set(range(self.n))
        new_cycle = []
        curr_term = []
        start_new = True
        while remain: # (0 2 3)(1 2)(3)(3 2)
            if start_new:
                #print("starting new term, remain is", remain)
                #print("cycle so far is", new_cycle)
                start_new = False
                elem = min(remain)
                remain.remove(elem)
                curr_term.append(elem)
            for term in self.cycle:
                #print("Term iis ", term)
                term_size = len(term)
                try:
                    loc = term.index(elem)
                    #print(elem, "goes to", end=" ")
                    elem = term[(loc+1) % term_size]
                    #print(elem, f"({term=})")
                except ValueError:
                    continue
            if elem == curr_term[0]:
                #print(f"finished cycle {curr_term=}")
                start_new = True
                new_cycle.append(curr_term.copy())
                curr_term = []
            else:
                curr_term.append(elem)
                remain.remove(elem)
        #print("simplified cycle", new_cycle)
        new_cycle.append(curr_term.copy())
        self.cycle = [x for x in new_cycle if len(x) > 1]

    def inv(self):
        return Permutation(cycle_notation=list(reversed([list(reversed(x)) for x in self.cycle])), n=self.n)

    def __mul__(self, other):
        inferred_n = max(self.n, other.n)
        cycle = self.cycle + other.cycle
        return Permutation(cycle_notation=cycle, n=inferred_n)

    def __hash__(self):
        return hash(str(self))

    def __truediv__(self, other):
        return self * other.inv()

    def __eq__(self, other):
        return str(self) == str(other)



pt1 = Permutation(cycle_notation=[], n=4)
pt2 = Permutation(cycle_notation=[[0,2,1],[1,2],[3],[3,2]])
pt3 = Permutation(result_notation=[1,4,2,0,3])

# (0 2 3)(1 2)(3)(3 2)
#print(pt1 * pt2)
#print(pt1, pt2, pt3)
#print(pt1.inv(), pt2.inv(), pt3.inv(), pt1/pt2)
#print(pt2/pt3)
#print(pt3)

def get_all_permutations(n):
    return [Permutation(result_notation=pt) for pt in itertools.permutations(list(range(n)))]

#def get_all_permutations(n):
#    for pt in itertools.permutations(list(range(n))):
#        yield Permutation(result_notation=pt)

def conjugacy_class(pt):
    all_perms = get_all_permutations(pt.n)
    reachable = set()
    for perm in all_perms:
        #new_elem = perm*pt/perm
        #if new_elem not in reachable:
        #   print(perm, "generates", new_elem)
        reachable.add(perm * pt / perm)
    return reachable

def orbit(base_elem):
    elem = base_elem
    for i in range(elem.n):
        if elem.is_identity:
            return i+1
        elem = elem*base_elem

def generate_single(base_elem):
    reachable = set()
    elem = base_elem
    reachable.add(elem)
    while not elem.is_identity:
        elem = elem*base_elem
        reachable.add(elem)
    return reachable

def generate(elems):
    if not isinstance(elems, Iterable):
        elems = [elems]

    orbits = set()
    for elem in elems:
        if elem not in orbits:
            orbits |= generate_single(elem)
    reachable = set()
    for elem1 in orbits:
        for elem2 in orbits:
            reachable.add(elem1 * elem2)
    return reachable


def centralizer(elems):
    if not isinstance(elems, Iterable):
        elems = [elems]
    all_perms = get_all_permutations(max(elems, key=lambda x: x.n).n)
    commuters = set()
    for candidate in all_perms:
        for pt in elems:
            if pt*candidate != candidate*pt:
                #print("elem", candidate, "fails on", pt)
                #print("LHS", pt*candidate, "RHS", candidate*pt)
                break
        else:
            #print(commuters)
            commuters.add(candidate)
    return commuters


def normalizer(elems):
    if not isinstance(elems, Iterable):
        elems = generate(elems)
    elems = set(elems)
    all_perms = get_all_permutations(max(elems, key=lambda x: x.n).n)
    commuters = set()
    for candidate in all_perms:
        for elem in elems:
            if candidate*elem/candidate not in elems:
                break
        else:
            commuters.add(candidate)
    return commuters

def center(n):
    return centralizer(get_all_permutations(n))


#pt_bad = Permutation([[0,3,2]], n=4)
#print(pt_bad, pt2)
#print(pt_bad.inv())
#print(pt_bad * pt2)
#print(pt_bad/pt2)
#print(pt_bad *pt2 /pt_bad)
pt4 = Permutation([[1,3,0,2]])
pt5 = Permutation([[0,1]], n=4)
#print(pt4 == pt2)
#print(pt5.cycle_type)
#print(pt5)
#print(conjugacy_class(pt5))
#print(centralizer(pt5))
#print(centralizer(Permutation([[0, 1], [2, 3]])))
#print(pt4, generate(pt4))
#print(normalizer(pt5))
#print(center(4), len(center(4)))
#print(orbit(pt5), orbit(pt4), orbit(pt2), orbit(pt3))
#print(Permutation(result_notation=[0,1,3,2]))

#gr = Group([(("r", 8), "e"),
#            (("f", 2), "e"),
#            ([("r", 1), ("f", 1), ("r", 1)], ("f", 1)),
#            ([("f", 1), ("r", 1)], [("r", 7), ("f", 1)])
#           ])
gr = Group(["r8 = e",
            "f2 = e",
            "r f r = f",
            "f r = r7 f",
            "f r2 = r6 f"])
#print(gr.parse("r25 f1 r2 r4 r-17 f1 f2 t12"))

#pe1 = gr.parse(e1)
#print("#"*50)
#print(Expression([], gr) * pe1)
#print("#"*50)
#print(Expression([gr.identity()], gr) * pe1)
#print("#"*50)
#quit()
e0 = "e e e e f e e e e e e e f e e e"
e1 = "r25 f1 r2 r4 r-17 f1 f2 t12"
e2 = "e"
e3 = "1"
e4 = "f2 r8 e"
e5 = "r1"
e6 = "r2 f"
e7 = "r3 f e r2 f e f"
e8 = "r9 f r r r r"
e9 = "r r r2 f r3 f f r5 r4 r"
for v in [e0, e1, e2, e3, e4, e5, e6, e7, e8, e9]:
#for v in [e7]:
    print(v, "-->", gr.parse(v), "-->", gr.parse(v).simplify())



#A4_group = Group(["a3=e",
#                  "x2=e",
#                  "b3=e",
#                  "c3=e",
#                  "d3=e",
#                  "z2=e",
#                  "y2=e",
#                  "x a = b",
#                  "a x = c",
#                  "d2 a = z",
#                  "b2 a = y"])
#
#print(A4_group.parse("a x x b2 y z-1").simplify())

Q8_group = Group(["i2=n",
                  "j2=n",
                  "k2=n",
                  "n2=e",
                  "ij=k",
                  "in=ni",
                  "jn=nj"
                  "kn=nk"])

print(Q8_group.parse("i j j j i k j i k j i k j i k j i k j i k j i k j i k i j k i j k j i k j i k j k").simplify())
