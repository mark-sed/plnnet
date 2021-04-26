import random

size = 1000
train = ""
outputs = ""

for _ in range(size):
    x1 = random.random()
    x2 = random.random()
    x3 = random.random()
    train += "[{:.2f}, {:.2f}, {:.2f}].\n".format(x1, x2, x3)
    outputs += "[{:.2f}].\n".format(x1)

with open('train', "w") as f:
    f.write(train)

with open('outputs', "w") as f:
    f.write(outputs)
