def edit_distance(source, target):
    m = len(source)
    n = len(target)
    
    # Create a DP table to store the results of subproblems
    dp = [[0 for _ in range(n+1)] for _ in range(m+1)]
    
    # Initialize base cases (when one string is empty)
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
    
    # Fill the DP table using the recurrence relation
    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = 0 if source[i-1] == target[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,   # Deletion
                dp[i][j-1] + 1,   # Insertion
                dp[i-1][j-1] + cost  # Substitution
            )
    
    # Return the minimum edit distance (final value in the DP table)
    return dp[m][n]

# Test the function
source_word = "kitten"
target_word = "sitting"
print(f"The Minimum Edit Distance is: {edit_distance(source_word, target_word)}")
