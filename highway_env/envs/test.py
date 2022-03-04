
a = [('a','b'),('c','d')]

g = [item for item in a[0]]
print(g)

groups = {}

groups[1] = [0,1]

groups[1].append(2)
groups[2] = [0]
print(len(groups))