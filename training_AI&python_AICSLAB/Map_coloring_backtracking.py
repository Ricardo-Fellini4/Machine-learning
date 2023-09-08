def isSafe(graph, color):

	# check for every edge
	for i in range(6):
		for j in range(i + 1, 6):
			if (graph[i][j] and color[j] == color[i]):
				return False
	return True




def graphColoring(graph, m, i, color):

	# if current index reached end
        if (i == 6):

            # if coloring is safe
            if (isSafe(graph, color)):

                # Print the solution
                printSolution(color)
                return True
            return False

        # Assign each color from 1 to m
        for j in range(1, m + 1):
            color[i] = j

            # Recur of the rest vertices
            if (graphColoring(graph, m, i + 1, color)):
                return True
            color[i] = 0
        else:
            return False




def printSolution(color):
	print("Solution Exists:" " Following are the assigned colors ")
	for i in range(6):
		print(color[i], end=" ")


# Driver code

	# /* Create following graph and
	# test whether it is 3 colorable
    # variables WA,NT,Q,NSW,V,SA,T
graph = [
	[0, 1, 0, 0, 0, 1],
	[1, 0, 1, 0, 0, 1],
	[0, 1, 0, 1, 0, 1],
	[0, 0, 1, 0, 1, 1],
	[0, 0, 0, 1, 0, 1],
	[1, 1, 1, 1, 1, 0],
]
m = 3 # Number of colors

	
color = [0 for i in range(6)]

	# Function call
if (not graphColoring(graph, m, 0, color)):
	print("Solution does not exist")

