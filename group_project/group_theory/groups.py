from collections import defaultdict, deque
from typing import Union
from itertools import islice, repeat
from . import permutation

def sliding_window(iterable, n):  # standard itertools recipe
    it = iter(iterable)
    window = deque(islice(it, n), maxlen=n)  # maxlen => the first element of the window
    if len(window) == n:  # will be removed when we do the append operation
        yield tuple(window)
    for x in it:
        window.append(x)
        yield tuple(window)


IDENTITY_SYMBOLS = ["e", "1"]


class Term():  # a single instance of something like "r^3", r is the sym, 3 is the exp
    def __init__(self, sym, exp, group):
        self.sym = sym
        self.exp = exp
        self.group = group
        self.cyclic_rule = group.singleton_rules.get(sym, None)  # mostly for efficiency
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
    
    # LHS takes precendence, for Term * {Term, Expression}, backend multiplication to match the Expression API
    def _mul(self, other) -> Union["Expression", "Term"]:
        if self.is_identity:
            return other
        
        if isinstance(other, Term):  # Term * Term multiplication
            if other.is_identity:  # to avoid NoneType issues
                return self
            if self.sym == other.sym:
                return Term(self.sym, self.exp + other.exp, self.group)
            else:
                return Expression([self, other], self.group)
        
        elif isinstance(other, Expression):  # Term * Expression multiplication
            return Expression([self], self.group)._mul(other)
        else:
            raise NotImplementedError(f"Don't know how to multiply Term * {type(other)}")
        
    def copy(self):
        return Term(self.sym, self.exp, self.group)
        
    # frontend of multiplication
    def __mul__(self, other: Union["Expression", "Term"]) -> Union["Expression", "Term"]:
        if isinstance(other, Expression) or isinstance(other, Term):
            return self._mul(other)
        else:
            return NotImplemented

    def __repr__(self):
        if self.exp == 1:
            return self.sym
        if self.is_identity:
            return "e"
        return f"{self.sym}{self.exp}"
    
    # def __len__(self):  # to be consistent with Expression, but shouldn't ever get called
    #     return 1
    
    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return self.sym == other.sym and self.exp == other.exp
    
    def __ge__(self, other):
        if self.is_identity:
            return other.is_identity
        return self.sym == other.sym and self.exp >= other.exp

    def inv(self):
        if self.is_identity:
            return self
        return Term(self.sym, -self.exp, self.group)
    
    def __pow__(self, other):
        return Term(self.sym, self.exp*other).simplify()

    # backend of division (no simplify step)
    def _truediv(self, other):
        return self._mul(other.inv())
    
    # frontend division (yes simplify step)
    def __truediv__(self, other):
        if isinstance(other, Expression) or isinstance(other, Term):
            return self._mul(other.inv()).simplify()
        else:
            return NotImplemented



class Expression():
    def __init__(self, expr: list[Term], group: "Group"):
        self.group = group
        self.expr = expr
        if not self.expr:
            self.expr = [group._identity_term()]  # allowing empty expresions makes things buggy


    def windows_match(self, window, pattern):
        if (window[0] >= pattern[0]) and (window[-1] >= pattern[-1]):
            for actual,expected in zip(window[1:-1], pattern[1:-1]):
                if actual != expected:
                    return False
            return True
        return False

    def tprint(self, *args, **kwargs):
        if self.group.verbose:
            print(*args, **kwargs)

    def simplify(self, max_iters=200) -> "Expression":
        updated = True  # if we've applied a rule or not in the given iteration
        n = 0  # check total iterations so far
        self.tprint("doing the printing?")
        starting = self.copy()  # use this to compare against for simplify_cache

        while updated and n < max_iters:
            n += 1
            self.tprint("curr expr is", self)
            updated = False

            self = self._filter_identity()  # first step of simplificiation is eliminating identities

            if self in self.group.simplify_cache:  # try exiting early
                self = self.group.simplify_cache[self].copy()
                break

            for window_size in sorted(self.group.general_rules.keys(), reverse=True):  # check all possible window-sizes we have rules for
                if len(self) < window_size:
                    continue
                new_expr = self.group._identity_expr()

                window_iter = sliding_window(self.expr, window_size)  # current window we are looking to apply rules in
                last_posn = 0  # track where we've appended up to, so we can append missing ones at the end
                for window in window_iter:
                    self.tprint("checking window of", window, type(window))
                    for pattern,result in self.group.general_rules[window_size]:  # check all possible rules at this given window
                        self.tprint("\tchecking window against", pattern)
                        if self.windows_match(window, pattern):  # if the rule applies
                            self.tprint("\t\twindow matches, proceeding with replacement...")
                            updated = True
                            self.tprint("\t\tbefore replacing, new_expr is now", type(new_expr), new_expr)
                            
                            # the result of applying the rule.
                            # filter identity to save a bit of compute (calculating translation doesn't filter for it
                            # ie. window[0]/pattern[0] * result * window[-1]/pattern[-1]

                            if window_size == 2 and len(result) == 2:
                                # very specific optimization for special type of rules
                                # based on the idea that if `s r = r^k s`, then `s^m r^l = r^(l*k^m) s^m`
                                left_exponent = window[0].exp  # m 
                                right_exponent = window[1].exp  # l
                                result_exp = result[0].exp  # k
                                # do modular exponentiation for a speed-up
                                new_exp = right_exponent*pow(result_exp, left_exponent, self.group.singleton_rules.get(result[0].sym))
                                translation = Expression([Term(result[0].sym, new_exp, self.group),
                                                          Term(result[1].sym, left_exponent, self.group)], self.group)
                            elif window_size > 1:
                                translation = result._concat([window[0]._truediv(pattern[0])], [window[-1]._truediv(pattern[-1])])
                            else:  # window_size == 1
                                translation = result._concat([], [window[-1]._truediv(pattern[-1])])
                            
                            self.tprint("\t\twill be adding", translation)

                            new_expr = new_expr._mul(translation)
                            self.tprint("\t\tafter replacing, new_expr is now", new_expr)
                            self.tprint("\t\twindow matched, advancing window by", window_size, "spaces")
                            try:  # skip window_size worth of windows because we've already used all those terms
                                last_posn += window_size
                                self.tprint("\t\tnew last posn", last_posn)
                                for _ in range(window_size-1):
                                    window = next(window_iter)
                                #window = [next(window_iter) for _ in range(window_size)][0]
                            except StopIteration:
                                self.tprint("stop iter")
                                break # no need to check other rules if we've ran out of windows to look at
                            self.tprint("\t\tafter advancing, new_expr is now", new_expr)
                            break  # we've made a valid pattern match at this location, so don't check any more patterns
                    else:  # if we reach the end, we've checked all patterns and nothing worked, so just append 1 term and move the window
                        self.tprint("\tbefore appending, new_expr is now", new_expr)
                        new_expr = new_expr._mul(window[0])
                        last_posn += 1
                        self.tprint("\tafter appending, new_expr is now", new_expr)

                self.tprint("appending last window of", self[last_posn:])
                if last_posn != len(self): # append any missing terms that got skipped over because we moved window
                    new_expr = new_expr._mul(self[last_posn:])
                self.tprint("end_cycle new_expr is now", new_expr)
                self = new_expr
        self.tprint("n is", n, max_iters)
        self.group.simplify_cache[starting] = self
        return self

    def _combine_like_terms(self, n: int) -> "Expression":
        curr_term = self.group._identity_term()
        for i, (term1, term2) in enumerate(zip(self.expr[n-1::-1], self.expr[n:])):
            #print("looking at", term1, curr_term, term2)
            curr_term = term1._mul(curr_term) #  * term2
            curr_term = curr_term._mul(term2)
            if isinstance(curr_term, Expression):
                break
        else:
            curr_term = [curr_term]
        curr_term = self._filter_identity(curr_term)
        new_expr = curr_term._concat(self.expr[:n-i-1], self.expr[n+i+1:]) # self.expr[:n-i-1] * curr_term *  self.expr[n+i+1:]
        
        if isinstance(new_expr, Expression):
            return new_expr
        self.expr = new_expr  # can modify in-place here since only used via __mul__, which creates a new Expression anyway
        return self

    def _filter_identity(self, expr=None) -> "Expression":  # remove all identity terms from an Expression or a list
        if expr is None:
            expr = self.expr
        if isinstance(expr, Expression):
            expr_terms = expr.expr
        else:
            expr_terms = expr
        return Expression(list(filter(lambda x: not x.is_identity, expr_terms)), self.group)
    
    @property
    def is_identity(self):
        return all([x.is_identity for x in self.expr])

    def __getitem__(self, idx) -> Term:
        return self.expr[idx]

    def __len__(self):
        return len(self.expr)

    # the pupose of this methods is to multiply 2 Expressions or lists such that combine_like_terms
    # isn't called. Used to avoid infinite loops, but probably shouldn't be used by clients. Should only use
    # when you know the combining like terms won't do anything
    # for multiplying {list,Expression} * Expression * {list, Expression}
    def _concat(self, left: Union[list, "Expression"], 
                right: Union[list, "Expression"]) -> "Expression": 
        if isinstance(left, list):
            left = Expression(left, self.group)
        if isinstance(right, list):
            right = Expression(right, self.group)
        return Expression(left.expr + self.expr + right.expr, self.group)._filter_identity()
    
    def copy(self):
        return Expression([t.copy() for t in self.expr], self.group)

    # for doing multiply in a simplify operatino so that we don't infinitely recurse
    def _mul(self, other) -> "Expression": # for Expression * {Expression, Term}        
        if isinstance(other, Expression):
            other_expr = other.expr
        elif isinstance(other, list):
            other_expr = other
        elif isinstance(other, Term):
            other_expr = [other]
        else:
            raise NotImplementedError(f"Don't know how to multiply Expression * {type(other)}")
        new_expr = Expression(self.expr + other_expr, self.group)
        return new_expr._combine_like_terms(len(self))

    
    # frontend of multiplication, so that simplification is done automatically
    def __mul__(self, other):  # Expression * {Expression, Term}
        if isinstance(other, Expression) or isinstance(other, Term):
            return self._mul(other).simplify()
        else:
            return NotImplemented
    
    def inv(self):
        if self.is_identity:
            return self
        return Expression([x.inv() for x in self.expr[::-1]], self.group)

    # backend of division (no simplify step)
    def _truediv(self, other):
        return self._mul(other.inv())
    
    # frontend division (with simplify step)
    def __truediv__(self, other):
        if isinstance(other, Expression) or isinstance(other, Term):
            return self._mul(other.inv()).simplify()
        else:
            return NotImplemented
    
    def __pow__(self, other):
        return Expression(list(repeat(self.expr, other)), self.group)

    def __eq__(self, other):  # need to check len because zip truncates elements
        try:
            return len(self) == len(other) and all(t1 == t2 for t1,t2 in zip(self.expr, other.expr))
        except TypeError:
            print(self, type(self), "######", other, type(other))
            raise

    def __repr__(self):
        return " ".join([str(t) for t in self.expr])
    
    def __hash__(self):
        return hash(str(self))


class Group(set):  # includes both the elements of the Group and the rules of the representation
    def __init__(self, *elems, rules=None, generate=False, verbose=False):
        super().__init__(elems)
        self.singleton_rules = {}
        self.general_rules = defaultdict(list)
        self.symbols = set()
        self.simplify_cache = {}
        self.verbose = verbose

        if rules:
            for rule in rules:
                pattern, result = rule.split("=")
                pattern_expr = self.parse(pattern)
                result_expr = self.parse(result)
                
                self._add_syms(pattern_expr, result_expr)  # so that we can generate the group later

                if len(pattern_expr) == 1 and  result_expr.is_identity:  # if symbol is cyclic, do this for efficiency
                    self.singleton_rules[pattern_expr[0].sym] = pattern_expr[0].exp
                    continue
                self.general_rules[len(pattern_expr)].append((pattern_expr, result_expr))    # map symbol -> (exponent, replacement)
        
        if generate:  # should probably only use this for finite groups
            self._generate_all()  # can't really handle infinite groups yet

        if elems and isinstance(elems[0], permutation.Permutation):
            self.n = elems[0].n

    def _generate_all(self):  # generate all elements in the group
        elems = self.generate(*self.symbols)
        self |= elems

    def generate(self, *exprs) -> "Group":
        if len(exprs) == 0:
            return self._identity_group()
    
        if isinstance(exprs[0], str):
            exprs = [self.parse(expr) for expr in exprs]

        frontier = self.subgroup(*exprs)
        visited = self.subgroup()
        while len(frontier) > 0:  # BFS
            start = frontier.pop()
            #print("checking elem", start)
            for elem in exprs:
                next = start*elem
                if next not in visited:
                    #print("found new node", next)
                    frontier.add(next)
                    visited.add(next)
                    #yield next  # so that we can do infinite groups as well
        return visited
    
    @property
    def is_perm_group(self):
        return hasattr(self, "n")  # permutation groups have n defined, others don't
        
    def _identity_term(self): # helper function to return an identity element, should only be called internally
        return Term(None, None, self)  # clients only use is_identity, and is_identity needs to be implemented everywhere
    
    def _identity_expr(self):  # helper function to return an expr, pretty much only used in generate
        if self.is_perm_group:
            return permutation.Permutation(None, n=self.n)
        else:
            return Expression([self._identity_term()], self)
    
    def _identity_group(self):   # helper function return a Group containing only an identity {Expression, Permutation}
        expr = self._identity_expr()
        return self.subgroup(expr)
    
    def subgroup(self, *elems):  # create an empty subgroup that has the same multiplication rules
        group = Group(*elems)
        set_these = ["singleton_rules", "general_rules", "n", "symbols", "verbose", "simplify_cache"]
        for var_name in set_these:
            if hasattr(self, var_name):
                setattr(group, var_name, getattr(self, var_name))
        return group
    
    def _add_syms(self, *exprs):
        for expr in exprs:
            for term in expr:
                if not term.is_identity:
                    self.symbols.add(term.sym)

    def parse(self, equation) -> "Expression":  # turn a written string into an Expression
        terms = equation.strip().split()
        start = self._identity_term()
        for t in terms:
            if t[0] in IDENTITY_SYMBOLS:
                next_term = self._identity_term()
            elif len(t) == 1:
                next_term = Term(t[0], 1, self)  # 1 is default exponent
            else:
                next_term = Term(t[0], int(t[1:]), self)
            start = start._mul(next_term)
        if isinstance(start, Term):  # always return an Expression 
            return Expression([start], self)
        return start
    
    def evaluate(self, equation):
        return self.parse(equation).simplify()

    def copy(self):
        return self.subgroup(*self)

    def __mul__(self, other):  # ie Group * Term (right cosets)
        new_elems = self.subgroup()
        for elem in self:
            new_elems.add(elem * other)
        return new_elems
    
    def __rmul__(self, other): # ie. Term * Group (left cosets)
        try:
            new_elems = self.subgroup()
        except AttributeError:
            print(type(self), type(other))
            raise
        for elem in self:
            new_elems.add(other * elem)
        return new_elems
    
    def __truediv__(self, other): # ie. Group / {Term, Permutation}
        return self*other.inv()