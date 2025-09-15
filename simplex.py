import numpy as np

def simplex(c, A, b):
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
        A tuple containing the optimal value of the objective function and the optimal solution vector.
    """
    # Get the number of constraints and variables
    num_constraints, num_variables = A.shape

    # Create the initial simplex tableau
    tableau = np.zeros((num_constraints + 1, num_variables + num_constraints + 1))
    tableau[:-1, :num_variables] = A
    tableau[:-1, num_variables:-1] = np.eye(num_constraints)
    tableau[:-1, -1] = b
    tableau[-1, :num_variables] = -c

    while np.any(tableau[-1, :-1] < 0):
        # Bland's Rule for entering variable: choose the smallest index among negative coefficients in the last row
        negative_coeffs_indices = np.where(tableau[-1, :-1] < 0)[0]
        # If no negative coefficients, the loop condition will handle it, but for safety:
        if len(negative_coeffs_indices) == 0:
            break # Should not happen if while condition is true
        pivot_col = negative_coeffs_indices[0]

        # Find the pivot row
        pivot_col_values = tableau[:-1, pivot_col]

        # Check for unboundedness
        if not np.any(pivot_col_values > 0):
            return "unbounded", None

        ratios = np.full(num_constraints, np.inf)
        positive_mask = pivot_col_values > 0
        ratios[positive_mask] = tableau[:-1, -1][positive_mask] / pivot_col_values[positive_mask]

        # Bland's Rule for leaving variable: choose the smallest index among those tied for the minimum ratio
        min_ratio = np.min(ratios)
        min_ratio_indices = np.where(ratios == min_ratio)[0]
        pivot_row = min_ratio_indices[0]

        # Perform the pivot operation
        pivot_element = tableau[pivot_row, pivot_col]
        tableau[pivot_row, :] /= pivot_element
        for i in range(num_constraints + 1):
            if i != pivot_row:
                tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]

    # Check for infeasibility: if the tableau is optimal but the solution is not feasible
    # (i.e., some basic variables are negative), then the problem is infeasible.
    if np.any(tableau[:-1, -1] < -1e-9):
        return "infeasible", None

    # Extract the solution
    solution = np.zeros(num_variables)
    for i in range(num_variables):
        col = tableau[:-1, i]
        if np.sum(col) == 1 and len(col[col == 1]) == 1:
            row_index = np.where(col == 1)[0][0]
            solution[i] = tableau[row_index, -1]

    optimal_value = tableau[-1, -1]
    return optimal_value, solution

if __name__ == '__main__':
    print("Simplex Algorithm Solver")
    print("="*25)
    print("Enter the linear programming problem in standard form:")
    print("Maximize Z = c^T * x")
    print("Subject to: Ax <= b")
    print("\nNote: The non-negativity constraints (x >= 0) are assumed")
    print("and do not need to be entered.")
    print("="*25)

    while True: # Main loop to solve multiple problems
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
                c_str = input(f"Enter the {num_variables} coefficients of the objective function (space-separated): ")
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

        print("Enter the coefficients of the constraints (A) and the right-hand side (b):")
        for i in range(num_constraints):
            while True:
                try:
                    constraint_str = input(
                        f"Constraint {i+1} ({num_variables} coefficients "
                        "and 1 RHS value, space-separated): "
                    )
                    constraint_vals = [float(x) for x in constraint_str.split()]
                    if len(constraint_vals) == num_variables + 1:
                        A[i, :] = constraint_vals[:num_variables]
                        b[i] = constraint_vals[num_variables]
                        break
                    else:
                        print(f"Please enter exactly {num_variables + 1} values.")
                except ValueError:
                    print("Invalid input. Please enter numbers separated by spaces.")

        try:
            # The simplex function already maximizes c^T * x
            optimal_value, solution = simplex(c, A, b)
            
            if optimal_value == "unbounded":
                print("\n" + "="*25)
                print("The linear programming problem is unbounded and has no finite solution.")
                print("="*25)
            elif optimal_value == "infeasible":
                print("\n" + "="*25)
                print("The linear programming problem is infeasible and has no solution.")
                print("="*25)
            else:
                print("\n" + "="*25)
                print("Solution (Maximization):")
                print(f"Optimal value: {optimal_value:.4f}")
                for i, val in enumerate(solution):
                    print(f"  x{i+1} = {val:.4f}")
                print("="*25)
        except Exception as e:
            print(f"\nAn error occurred: {e}")

        while True:
            another = input("\nDo you want to solve another problem? (yes/no): ").lower()
            if another in ["yes", "no"]:
                break
            else:
                print("Invalid input. Please enter 'yes' or 'no'.")

        if another == "no":
            break


