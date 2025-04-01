def edit_distance(source, target):
    m = len(source)
    n = len(target)
    dp = [[0 for _ in range(n+1)] for _ in range(m+1)]

    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j

    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = 0 if source[i-1] == target[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)

    return dp[m][n]

def correct_spelling(word, dictionary):
    min_distance = float('inf')
    corrected_word = ""
    for correct_word in dictionary:
        distance = edit_distance(word, correct_word)
        if distance < min_distance:
            min_distance = distance
            corrected_word = correct_word
    return corrected_word

# Test the spelling correction system
dictionary = ["cat", "bat", "rat", "mat", "kitten", "sitting"]
misspelled_word = "bass"
print(f"Corrected word: {correct_spelling(misspelled_word, dictionary)}")
