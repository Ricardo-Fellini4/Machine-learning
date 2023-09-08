# follows description in ch. 6.2.2 in textbook
def AC3(csp):
    queue = deque()
    # add all arcs to queue
    for constraint in csp.binaryConstraints:
        queue.append((constraint.var1, constraint.var2))
        queue.append((constraint.var2, constraint.var1))

    while len(queue) > 0:
        (i, j) = queue.popleft()
        if removeInconsistentValues(csp, i, j):
            if len(csp.varDomains[i]) == 0:
                return False
            for k in csp.getNeighbors(i):
                queue.append((k, i))
    return True

# returns true if we remove a value
def removeInconsistentValues(csp, i, j):
    removed = False
    # must use list() to iterate over copy of list
    # because we are changing the list as we iterate
    for x in list(csp.varDomains[i]):
        # if no value y in Dj allows (x,y) to satisfy the constraint between i and j
        if not hasSupport(csp, i, j, x):
            csp.varDomains[i].remove(x)
            removed = True
    return removed

# returns true if there is a value y in Dj that allows (x,y) to satisfy the constraint between i and j
def hasSupport(csp, i, j, x):
    for y in csp.varDomains[j]:
        if csp.constraintsAreSatisfied(i, x, j, y):
            return True
    return False
