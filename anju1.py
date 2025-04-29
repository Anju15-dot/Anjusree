from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import DiscreteDistribution
from pgmpy.factors.discrete import ConditionalProbabilityTable
from pgmpy.inference import VariableElimination

model = BayesianNetwork([('A', 'B'), ('B', 'C')])

cpd_A = DiscreteDistribution({'True': 0.6, 'False': 0.4})

cpd_B = ConditionalProbabilityTable(
    [['True', 'True', 0.9],
     ['True', 'False', 0.1],
     ['False', 'True', 0.7],
     ['False', 'False', 0.3]], [cpd_A])

cpd_C = ConditionalProbabilityTable(
    [['True', 'True', 0.8],
     ['True', 'False', 0.2],
     ['False', 'True', 0.6],
     ['False', 'False', 0.4]], [cpd_B])

model.add_cpds(cpd_A, cpd_B, cpd_C)

model.check_model()

inference = VariableElimination(model)

prob_A = inference.query(variables=['A'])
print("P(A):")
print(prob_A)

prob_B_given_A = inference.query(variables=['B'], evidence={'A': 'True'})
print("\nP(B | A=True):")
print(prob_B_given_A)

prob_C_given_B = inference.query(variables=['C'], evidence={'B': 'True'})
print("\nP(C | B=True):")
print(prob_C_given_B)
