import random

import numpy as np
import matplotlib.pyplot as plt


population = 500
mutations = 0.005
img = np.random.randint(2, size=(15, 15))
plt.imshow(img, cmap=plt.cm.gray)  # use appropriate colormap here
plt.show()


def ga(array, population, mutations):
    def score(matrix1, matrix2):
        return (matrix1 == matrix2).sum()

    rows = array.shape[0]
    columns = array.shape[1]
    mid = rows // 2

    mem = np.random.randint(2, size=(2 * population, rows, columns))
    scores = np.zeros((2 * population))
    bottom = list(range(len(mem)))

    for i in range(1000000):
        # Bottom will contain all the random individuals generated when starting the execution
        # and the new individuals after the first iteration. Bottom means the bottom of the list
        # sorted by score
        for k in bottom:
            scores[k] = score(mem[k], array)

        # Check if the solution has been found
        max_score = np.argmax(scores)
        if scores[max_score] == rows * columns:
            print(i)
            plt.imshow(
                mem[max_score], cmap=plt.cm.gray
            )  # use appropriate colormap here
            plt.show()
            break

        # Select the population of individuals according to the score function
        top_n_scores = np.argpartition(scores, population)
        top = top_n_scores[population:]
        bottom = top_n_scores[:population]

        # Create #population new elements from the crossover and mutation
        for j in range(population):
            # Crossover -> Select parents from the top individuals
            #
            # I tried this with random choice and just picking a random position
            # from the top and the next one and the result is the same but way faster
            # It might be because of either the randomization of the initial population or maybe
            # the implementation of argpartition? or both?
            r = random.randrange(len(top))
            idx = [r, (r + 1) % len(top)]
            parents = [top[idx[0]], top[idx[1]]]

            mem[bottom[j]][0:mid] = mem[parents[0]][0:mid]
            mem[bottom[j]][-(mid + 1) :] = mem[parents[1]][-(mid + 1) :]

            # Mutation -> Mutate the bits
            #
            # The random choice of the bits to mutate is the most costly of the implementation
            # It seems there has to be some way to speed up this
            idx = np.random.choice(
                [0, 1], p=[(1 - mutations), mutations], size=(rows, columns)
            )
            mem[bottom[j]] = abs(mem[bottom[j]] - idx)


ga(img, population, mutations)
