import numpy as np


def _phase1(c, A, b, constraint_types):
    num_constraints, num_variables = A.shape

    # Add slack, surplus, and artificial variables
    num_slack = constraint_types.count('<=')
    num_surplus = constraint_types.count('>=')
    num_artificial = num_surplus + constraint_types.count('==')

    tableau = np.zeros((num_constraints + 1, num_variables + num_slack + num_surplus + num_artificial + 1))

    # Fill the tableau
    tableau[:-1, :num_variables] = A
    tableau[:-1, -1] = b

    # Add slack, surplus, and artificial variables to the tableau
    s_idx = num_variables
    sur_idx = num_variables + num_slack
    art_idx = num_variables + num_slack + num_surplus

    artificial_var_cols = []
    for i in range(num_constraints):
        if constraint_types[i] == '<=':
            tableau[i, s_idx] = 1
            s_idx += 1
        elif constraint_types[i] == '>=':
            tableau[i, sur_idx] = -1
            tableau[i, art_idx] = 1
            artificial_var_cols.append(art_idx)
            sur_idx += 1
            art_idx += 1
        elif constraint_types[i] == '==':
            tableau[i, art_idx] = 1
            artificial_var_cols.append(art_idx)
            art_idx += 1

    # Set up the auxiliary objective function
    tableau[-1, artificial_var_cols] = 1
    for r in range(num_constraints):
        if tableau[r, artificial_var_cols].any():
            tableau[-1, :] -= tableau[r, :]

    # Solve the auxiliary problem
    optimal_val, _, tableau = _solve_simplex(tableau.copy(), num_variables + num_slack + num_surplus, num_constraints)

    # Check the result of Phase 1
    if not np.isclose(optimal_val, 0):
        return "infeasible", None, None

    # Prepare the tableau for Phase 2
    # Remove artificial variable columns and restore the original objective function
    tableau = np.delete(tableau, artificial_var_cols, axis=1)

    tableau[-1, :] = 0
    tableau[-1, :num_variables] = -c

    # Re-establish the basic feasible solution
    for i in range(num_constraints):
        # find basic variable in row i
        basic_var_col = -1
        for j in range(num_variables + num_slack + num_surplus):
            if np.isclose(tableau[i, j], 1) and np.count_nonzero(np.isclose(tableau[:-1, j], 0)) == num_constraints - 1:
                basic_var_col = j
                break
        if basic_var_col != -1:
            if not np.isclose(tableau[-1, basic_var_col], 0):
                tableau[-1, :] -= tableau[-1, basic_var_col] * tableau[i, :]

    return None, tableau, num_variables + num_slack + num_surplus


def simplex(c, A, b, constraint_types):
    num_constraints, num_variables = A.shape

    # Handle negative b values
    for i in range(num_constraints):
        if b[i] < 0:
            A[i, :] *= -1
            b[i] *= -1
            if constraint_types[i] == '<=':
                constraint_types[i] = '>='
            elif constraint_types[i] == '>=':
                constraint_types[i] = '<='


    # Check if we need to use the two-phase simplex method
    needs_two_phase = any(ct != ' <=' for ct in constraint_types) or np.any(b < 0)

    if not needs_two_phase:
        # Use the standard simplex method
        tableau = np.zeros((num_constraints + 1, num_variables + num_constraints + 1))
        tableau[:-1, :num_variables] = A
        tableau[:-1, num_variables:-1] = np.eye(num_constraints)
        tableau[:-1, -1] = b
        tableau[-1, :num_variables] = -c
        optimal_value, solution, _ = _solve_simplex(tableau, num_variables, num_constraints)
        return optimal_value, solution
    else:
        # Use the two-phase simplex method
        status, tableau, num_vars = _phase1(c, A, b, constraint_types)
        if status == "infeasible":
            return "infeasible", None

        optimal_value, solution, _ = _solve_simplex(tableau, num_vars, num_constraints)
        if solution is None:
            return optimal_value, None
        return optimal_value, solution[:num_variables]

def _solve_simplex(tableau, num_variables, num_constraints):
    """
    Solves a linear programming problem in standard form:
    Maximize Z = c^T * x
    Subject to: Ax <= b
                x >= 0

    Args:
        c: 1D numpy array of coefficients of the objective function.
        A: 2D numpy array of coefficients of the constraints.
        b: 1D numpy array of the right-hand side of the constraints.

    Returns:
        A tuple containing the optimal value of the objective function and the
        optimal solution vector.
    """
    # Get the number of constraints and variables
    """
    Solves a linear programming problem using the simplex algorithm on a given tableau.
    """
    while np.any(tableau[-1, :-1] < 0):
        # Bland's Rule for entering variable: choose the smallest index among negative coefficients in the last row
        negative_coeffs_indices = np.where(tableau[-1, :-1] < 0)[0]
        if len(negative_coeffs_indices) == 0:
            break
        pivot_col = negative_coeffs_indices[0]

        # Find the pivot row
        pivot_col_values = tableau[:-1, pivot_col]

        if not np.any(pivot_col_values > 0):
            return "unbounded", None, None

        ratios = np.full(num_constraints, np.inf)
        positive_mask = pivot_col_values > 0
        ratios[positive_mask] = (
            tableau[:-1, -1][positive_mask] / pivot_col_values[positive_mask]
        )

        min_ratio = np.min(ratios)
        min_ratio_indices = np.where(ratios == min_ratio)[0]
        pivot_row = min_ratio_indices[0]

        pivot_element = tableau[pivot_row, pivot_col]
        tableau[pivot_row, :] /= pivot_element
        for i in range(num_constraints + 1):
            if i != pivot_row:
                tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]

    if np.any(tableau[:-1, -1] < -1e-9):
        return "infeasible", None, None

    solution = np.zeros(num_variables)
    for i in range(num_variables):
        col = tableau[:-1, i]
        if np.sum(col) == 1 and len(col[col == 1]) == 1:
            row_index = np.where(col == 1)[0][0]
            solution[i] = tableau[row_index, -1]

    optimal_value = tableau[-1, -1]
    return optimal_value, solution, tableau


def main():
    print("Simplex Algorithm Solver")
    print("=" * 25)
    print("This program solves linear programming problems using the Simplex method.")
    print("You can define your objective function to maximize and specify various constraint types.")
    print("\nEnter your problem in the following format:")
    print("Maximize Z = c^T * x")
    print("Subject to: Ax [constraint_type] b")
    print("Where [constraint_type] can be '<=' (less than or equal to), '>=' (greater than or equal to), or '==' (equal to).")
    print("\nNote: Non-negativity constraints (x >= 0) are assumed and do not need to be entered.")
    print("=" * 25)

    while True:  # Main loop to solve multiple problems
        while True:
            try:
                num_variables = int(input("Enter the number of variables: "))
                if num_variables > 0:
                    break
                else:
                    print("Number of variables must be a positive integer.")
            except ValueError:
                print("Invalid input. Please enter an integer.")

        while True:
            try:
                c_str = input(
                    f"Enter the {num_variables} coefficients of the objective function "
                    "(space-separated): "
                )
                c = np.array([float(x) for x in c_str.split()])
                if len(c) == num_variables:
                    break
                else:
                    print(f"Please enter exactly {num_variables} coefficients.")
            except ValueError:
                print("Invalid input. Please enter numbers separated by spaces.")

        while True:
            try:
                num_constraints = int(input("Enter the number of constraints: "))
                if num_constraints > 0:
                    break
                else:
                    print("Number of constraints must be a positive integer.")
            except ValueError:
                print("Invalid input. Please enter an integer.")

        A = np.zeros((num_constraints, num_variables))
        b = np.zeros(num_constraints)
        constraint_types = []

        print(
            "Enter the coefficients of the constraints (A), the constraint type, and the right-hand side (b):"
        )
        for i in range(num_constraints):
            while True:
                try:
                    constraint_str = input(
                        f"Constraint {i+1} ({num_variables} coefficients, a type from ['<=', '>=', '=='], "
                        "and 1 RHS value, space-separated): "
                    )
                    constraint_vals = constraint_str.split()
                    if len(constraint_vals) == num_variables + 2:
                        A[i, :] = [float(x) for x in constraint_vals[:num_variables]]
                        constraint_types.append(constraint_vals[num_variables])
                        b[i] = float(constraint_vals[num_variables + 1])
                        break
                    else:
                        print(f"Please enter exactly {num_variables + 2} values.")
                except (ValueError, IndexError):
                    print("Invalid input. Please enter numbers and a valid constraint type separated by spaces.")

        try:
            # The simplex function already maximizes c^T * x
            optimal_value, solution = simplex(c, A, b, constraint_types)

            if optimal_value == "unbounded":
                print("\n" + "=" * 25)
                print(
                    "The linear programming problem is unbounded and has no finite "
                    "solution."
                )
                print("=" * 25)
            elif optimal_value == "infeasible":
                print("\n" + "=" * 25)
                print(
                    "The linear programming problem is infeasible and has no "
                    "solution."
                )
                print("=" * 25)
            else:
                print("\n" + "=" * 25)
                print("Solution (Maximization):")
                print(f"Optimal value: {optimal_value:.4f}")
                for i, val in enumerate(solution):
                    print(f"  x{i+1} = {val:.4f}")
                print("=" * 25)
        except Exception as e:
            print(f"\nAn error occurred: {e}")

        while True:
            another = input(
                "\nDo you want to solve another problem? (yes/no): "
            ).lower()
            if another in ["yes", "no"]:
                break
            else:
                print("Invalid input. Please enter 'yes' or 'no'.")

        if another == "no":
            break


if __name__ == "__main__":
    main()

