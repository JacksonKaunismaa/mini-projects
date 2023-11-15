def solve_puzzle(puzzle, moves_left):
    start, target, operations = puzzle

    # Base case: If the current number equals the target, return an empty sequence.
    if start == target:
        return []

    # Base case: If there are no moves left, return None to indicate no solution.
    if moves_left == 0:
        return None

    # Initialize the solution to None.
    solution = None

    for op_name, operation_func, is_possible_func in operations:
        if not is_possible_func(start):
            continue  # Skip this operation if it's not possible.

        new_num = operation_func(start)
        result = solve_puzzle((new_num, target, operations), moves_left - 1)

        if result is not None:
            # If a solution is found, append the current operation name to the result.
            solution = [op_name] + result
            break

    return solution

def delete_rightmost_digit(num):
    return num // 10

def mirror(num):
    # make sure to account for negatives
    sign = -1 if num < 0 else 1
    num = abs(num)
    return sign * int(str(num) + str(num)[::-1])

def rotate_left(num):
    return int(str(num)[1:] + str(num)[0])

def sum_all(num):
    return sum(int(digit) for digit in str(num))

func_names = {"mirror": mirror, "<<": delete_rightmost_digit, "shift<": rotate_left, "sum": sum_all}

# Function to create an operation tuple from an operation string
def create_operation_tuple(operation_str):
    if operation_str[0] == '*':
        op_name = f"*{operation_str[1:]}"
        operation_func = lambda x: x * int(operation_str[1:])
    elif operation_str[0] == '/':
        divisor = int(operation_str[1:])
        op_name = f"/{divisor}"
        operation_func = lambda x: x / divisor
    elif operation_str[0] == '+':
        op_name = f"+{operation_str[1:]}"
        operation_func = lambda x: x + int(operation_str[1:])
    elif operation_str[0] == '-':
        op_name = f"-{operation_str[1:]}"
        operation_func = lambda x: x - int(operation_str[1:])
    else:
        op_name = operation_str
        operation_func = func_names[operation_str]

    if operation_str == "mirror":
        is_possible_func = lambda x: True if len(str(x)) <= 3 else False
    elif operation_str == "/":
        is_possible_func = lambda x: True if x % divisor == 0 else False
    else:
        is_possible_func = lambda x: True

    return (op_name, operation_func, is_possible_func)

# List of operation strings
easy_ops = ["mirror", "sum"]
start, end, moves_left = 125, 20, 8

# Generate the operations list
operations = [create_operation_tuple(op) for op in easy_ops]

puzzle = (start, end, operations)
solution = solve_puzzle(puzzle, moves_left)
if solution:
    print("Solution found:", solution)
else:
    print("No solution.")
