from docplex.mp.model import Model
import numpy as np

# Initialisation step
b = np.array([97, 610, 395, 211])
w = [45, 36, 31, 14]
r = 100


def get_optiumum(A):
    model = Model()
    x_cst = []
    # add all variables in model to get y_i
    for i in range(len(A[0])):
        x_cst.append(model.continuous_var(name="x" + str(i), lb=0))

    # add constraints in model
    for i in range(4):
        cs = 0
        for j in range(len(A[0])):
            cs += A[i][j] * x_cst[j]
        # print(str(cs) + " = " + str(b[i]))
        model.add_constraint(cs >= b[i], ctname="const" + str(i))

    # resolve linear program
    model.set_objective("min", sum(x_cst))  # min because Ax <= b
    model.solve()
    # get dual
    dual = model.dual_values(model.iter_linear_constraints())
    # set up second linear program to get a_i
    second_model = Model()

    a_cst = []  # set up all a_i constraints, a_i >= 0
    for i in range(len(w)):
        a_cst.append(second_model.integer_var(name="a" + str(i), lb=0))
    cs = 0
    for i in range(len(w)):
        cs += w[i] * a_cst[i]

    second_model.add_constraint(cs <= r, ctname="const1")

    product_ai_yi = np.dot(a_cst, np.array(dual))
    second_model.set_objective("max", product_ai_yi)
    second_model.solve()

    list_solution = getList(second_model.solution.to_string(), a_cst)

    if second_model.objective_value > 1:
        # add model result to A matrix
        A = np.column_stack((A, list_solution))
        get_optiumum(A)
    else:
        pass
        print("Final matrixe: \n", A)


def getList(solution_to_string, a_cst):
    list_entree = list(solution_to_string.split("\n"))
    list_entree = list_entree[3:len(list_entree) - 1]
    a_cst = ["a" + str(i) for i in range(len(a_cst))]
    tab = np.zeros(len(a_cst), dtype=int)
    for element in list_entree:
        for a in range(len(a_cst)):
            if element.split("=")[0] == a_cst[a]:
                tab[a] = round(float(element.split("=")[1]))
    return tab


get_optiumum(np.identity(4, dtype=int))