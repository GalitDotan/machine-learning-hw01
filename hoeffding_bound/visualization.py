import matplotlib.pyplot as plt
import numpy

ROWS = 200000
COLS = 20
PROB = 0.5

NUM_EPSILONS = 50

matrix = numpy.random.binomial(n=1, p=PROB, size=(ROWS, COLS))
empirical_means = numpy.mean(matrix, axis=1)
epsilons = numpy.linspace(start=0, stop=1, num=NUM_EPSILONS)

a = numpy.abs(empirical_means - PROB)
empirical_probabilities = [sum(a > e) / ROWS for e in epsilons]

hoeffding = 2 * numpy.exp(-2 * COLS * epsilons ** 2)

plt.plot(epsilons, empirical_probabilities, label='empirical probabilities')
plt.plot(epsilons, hoeffding, label='hoeffding bound')
plt.xlabel('Epsilon')
plt.legend()
plt.show()
