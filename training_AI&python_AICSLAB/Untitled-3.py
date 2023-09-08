
domains = [[1, 2, 3] for i in range(6)]
graph = [
	[0, 1, 0, 0, 0, 1],
	[1, 0, 1, 0, 0, 1],
	[0, 1, 0, 1, 0, 1],
	[0, 0, 1, 0, 1, 1],
	[0, 0, 0, 1, 0, 1],
	[1, 1, 1, 1, 1, 0],
]



def forward_checking(assignment, domains):
    # Make a copy of color
    new_domains =domains.copy()
    for i in new_domains:
        for j in new_domains[i]:
            for k in assignment:
                if graph[i][k] == 1:
                    if j == assignment[k]:
                        new_domains[i][j] = 0
                        if new_domains[i]==[0,0,0]:
                            return None
                        break
    return new_domains
def backtracking_search(assignment):
    if len(assignment) == 6 and all(v in domains[u] for u,v in assignment.item()): 
        return assignment
    var = [v for v in range(6) if( assignment[v] == 0)][0]
    for val in domains[var]:
        assignment[var]=val 
        new_domains =forward_checking(assignment,domains)
        if new_domains is not None:
            result = backtracking_search(assignment)
            if result is not None:
                return result
        del assignment[var]
    return None


assignment= []
solution = backtracking_search(assignment)
print(solution)



# assignment = [0 for i in range(6)]