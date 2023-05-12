import itertools
from . import groups

class Permutation():
    def __init__(self, cycle_notation=None, result_notation=None, n=None):
        assert (cycle_notation is None) ^ (result_notation is None)
        notation = cycle_notation if cycle_notation is not None else result_notation

        #print(cycle_notation,result_notation, notation)

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
    

    def identity(self):
        return Permutation([], n=self.n)

    @property
    def cycle_type(self):
        cycle_lens = [len(x) for x in self.cycle]
        num_cycles = []
        for i in range(self.n):  # the 1's count will likely be wrong
            num_cycles.append(cycle_lens.count(i+1))
        return num_cycles
    
    @property
    def parity(self):
        return sum(i*amt for i,amt in enumerate(self.cycle_type)) % 2 

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

    def __mul__(self, other):  # for Permutation * Permutation
        if isinstance(other, Permutation):
            #print(other, "hey", type(other), other, self, isinstance(other, Permutation))
            inferred_n = max(self.n, other.n)
            cycle = self.cycle + other.cycle
            return Permutation(cycle_notation=cycle, n=inferred_n)
        else:
            #print(other, "hoy", type(other), other, self, isinstance(other, Permutation))
            return NotImplemented            

    def __hash__(self): 
        return hash(str(self))

    def __truediv__(self, other):
        if isinstance(other, Permutation):
            return self * other.inv()
        else:
            return NotImplemented

    def __eq__(self, other):
        return str(self) == str(other)


def get_all_permutations(n):
    return groups.Group(*list(Permutation(result_notation=pt) for pt in itertools.permutations(list(range(n)))))

def get_even_permutations(n):
    return groups.Group(*list(filter(lambda x: x.parity == 0, get_all_permutations(n))))
