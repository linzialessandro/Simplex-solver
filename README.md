# Simplex Solver

A simple Python implementation of the Simplex algorithm for solving Linear Programming problems. This solver is designed to handle maximization problems in standard form.

## Features

*   Solves Linear Programming problems using the two-phase Simplex algorithm.
*   Supports maximization problems.
*   Handles constraints of the form `Ax <= b`, `Ax >= b`, and `Ax = b`.
*   Handles negative values in the right-hand side of the constraints.
*   Assumes non-negativity constraints (`x >= 0`).
*   Detects unbounded and infeasible problems.
*   Uses Bland's Rule for pivot selection to prevent cycling.
*   Interactive command-line interface for inputting problems.

## Getting Started

### Prerequisites

Make sure you have Python 3 and NumPy installed.

```bash
pip install numpy
```

### How to Run

1.  Clone this repository (or download `simplex.py`).
2.  Navigate to the project directory in your terminal.
3.  Run the script:

    ```bash
    python simplex.py
    ```

4.  Follow the on-screen prompts to enter your linear programming problem.

## Problem Input Format

The solver expects problems in the following standard form:

**Maximize:** `Z = c^T * x`
**Subject to:** `A * x [<, >, =] b`
**And:** `x >= 0` (non-negativity is assumed)

You will be prompted to enter:
*   The number of variables.
*   The coefficients of the objective function (`c`), space-separated.
*   The number of constraints.
*   For each constraint, the coefficients of `A`, the constraint type (`<=`, `>=`, or `==`), and the right-hand side `b` value, space-separated.

## Example

Consider the problem:

Maximize `Z = 3x1 + 2x2`
Subject to:
`x1 + x2 <= 4`
`2x1 + x2 <= 5`
`x1, x2 >= 0`

You would enter:

```
Enter the number of variables: 2
Enter the 2 coefficients of the objective function (space-separated): 3 2
Enter the number of constraints: 2
Constraint 1 (2 coefficients and 1 RHS value, space-separated): 1 1 4
Constraint 2 (2 coefficients and 1 RHS value, space-separated): 2 1 5
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
