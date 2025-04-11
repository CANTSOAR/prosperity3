rewards = [90, 89, 80, 73, 50, 37, 31, 20, 17, 10]
inhabitants = [10, 8, 6, 4, 4, 3, 2, 2, 1, 1]
weights = [10 for x in range(10)]

equal = False
while not equal:
    average = 0
    for x in range(10):
        average += rewards[x] / (inhabitants[x] + weights[x])
    average /= 10

    for x in range(10):
        if rewards[x] / (inhabitants[x] + weights[x]) < average:
            weights[x] -= .001
            for y in range(10):
                if y != x:
                    weights[y] += .001 / 9

    equal = True
    for x in range(10):
        if rewards[x] / (inhabitants[x] + weights[x]) - average > .001:
            equal = False


print(weights)
print(rewards[0] / (inhabitants[0] + weights[0]))