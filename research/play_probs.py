import itertools

results = []
ranks = ["3", "4", "5", "6", "7", "8", "9", "0", "J", "Q", "K"]

# Singles (Not A, 2)
# This is 1 minus the chance none of the 4 cards in the rank are in the hand.
temp = 1 - (32/48) * (31/47) * (30/46) * (29/45)
print("Probability of playing a single card (not A, 2):", temp)
for r in ranks:
    results.append((f"{r}", temp))
# Singles (A)
temp = 1 - (32/48) * (31/47) * (30/46)
print("Probability of playing an Ace:", temp)
results.append(("A", temp))
# Singles (2)
temp = 1 - (32/48)
print("Probability of playing a 2:", temp)
results.append(("2", temp))

# Doubles (Not A)
# This is 1 minus the chance of 0 or 1 cards in the rank.
temp = 1 - (32/48) * (31/47) * (30/46) * (29/45) - 4 * (16/48) * (32/47) * (31/46) * (30/45)
print("Probability of playing a double (not A):", temp)
for r in ranks:
    results.append((f"{r}{r}", temp))
# Doubles (A)
temp = 1 - (32/48) * (31/47) * (30/46) - 3 * (16/48) * (32/47) * (31/46)
print("Probability of playing a double (A):", temp)
results.append(("AA", temp))

# Triples
# This is the chance of 3 cards plus 4 cards in the rank
temp = 4 * (16/48) * (15/47) * (14/46) * (32/45) + (16/48) * (15/47) * (14/46) * (13/45)
print("Probability of playing a triple:", temp)
for r in ranks:
    results.append((f"{r}{r}{r}", temp))

# Full House (Not A in triple or double)
# Chance of triple
triple_chance = 4 * (16/48) * (15/47) * (14/46) * (32/45) * (1 - (31/44) * (30/43) * (29/42) * (28/41) - 4 * (13/44) * (31/43) * (30/42) * (29/41))
# Chance of quad
quad_chance = (16/48) * (15/47) * (14/46) * (13/45) * (1 - (32/44) * (31/43) * (30/42) * (29/41) - 4 * (12/44) * (32/43) * (31/42) * (30/41))
# Chance of full house
temp = triple_chance + quad_chance
print("Probability of playing a full house (not A):", temp)
for r in ranks:
    for r2 in ranks:
        if r != r2:
            results.append((f"{r}{r}{r}{r2}{r2}", temp))
# Full House (A in triple)
temp = (16/48) * (15/47) * (14/46) * (1 - (32/45) * (31/44) * (30/43) * (29/42) - 4 * (13/45) * (32/44) * (31/43) * (30/42))
print("Probability of playing a full house (A in triple):", temp)
for r in ranks:
    results.append((f"AAA{r}{r}", temp))
# Full House (A in double)
# Chance of a triple
temp = 4 * (16/48) * (15/47) * (14/46) * (32/45) * (1 - (31/44) * (30/43) * (29/42) - 3 * (13/44) * (31/43) * (30/42))
# Chance of a quad
quad_chance = (16/48) * (15/47) * (14/46) * (13/45) * (1 - (32/44) * (31/43) * (30/42) - 4 * (12/44) * (32/43) * (31/42))
# Chance of full house
temp = triple_chance + quad_chance
print("Probability of playing a full house (A in double):", temp)
for r in ranks:
    results.append((f"{r}{r}{r}AA", temp))

# Bomb (Not A, No extra)
temp = (16/48) * (15/47) * (14/46) * (13/45)
print("Probability of playing a bomb (not A, no extra):", temp)
for r in ranks:
    results.append((f"{r}{r}{r}{r}", temp))
# Bomb (Not A, not A, 2 extra)
temp = (16/48) * (15/47) * (14/46) * (13/45) * (1 - (32/44) * (31/43) * (30/42) * (29/41))
print("Probability of playing a bomb (Not A, not A, 2 extra):", temp)
for r in ranks:
    for r2 in ranks:
        if r != r2:
            results.append((f"{r}{r}{r}{r}{r2}", temp))
# Bomb (Not A, A extra)
temp = (16/48) * (15/47) * (14/46) * (13/45) * (1 - (32/44) * (31/43) * (30/42))
for r in ranks:
    results.append((f"{r}{r}{r}{r}A", temp))
# Bomb (Not A, 2 extra)
temp = (16/48) * (15/47) * (14/46) * (13/45) * (12/44)
print("Probability of playing a bomb (Not A, 2 extra):", temp)
for r in ranks:
    results.append((f"{r}{r}{r}{r}2", temp))
# Bomb (A, no extra)
temp = (16/48) * (15/47) * (14/46)
print("Probability of playing a bomb (A, no extra):", temp)
results.append(("AAA", temp))
# Bomb (A, not 2 extra)
temp = (16/48) * (15/47) * (14/46) * (1 - (32/45) * (31/44) * (30/43) * (29/42))
print("Probability of playing a bomb (A, not 2 extra):", temp)
for r in ranks:
    results.append((f"AAA{r}", temp))
# Bomb (A, 2 extra)
temp = (16/48) * (15/47) * (14/46) * (13/45)
print("Probability of playing a bomb (A, 2 extra):", temp)
results.append(("AAA2", temp))

def generate_valid_tuples(length, max_sum, min_value=1, current=(), current_sum=0):
    """Efficiently generates l-tuples of (1,2,3,4) whose sum does not exceed max_sum."""
    if len(current) == length:
        yield current
    else:
        for value in range(min_value, 5):
            new_sum = current_sum + value
            if new_sum > max_sum:
                continue  # Prune search early
            yield from generate_valid_tuples(length, max_sum, min_value, current + (value,), new_sum)

for l in range(5, 12):  # Straight lengths from 5 to 11
    total = 0
    for i in generate_valid_tuples(l, 16):  # Use backtracking generator
        temp = 1
        for j in i:
            if j == 1 or j == 3:
                temp *= 4
            elif j == 2:
                temp *= 6
        s = sum(i)
        for j in range(s):
            temp *= (16 - j) / (48 - j)
        for j in range(4 * l - s):
            temp *= (32 - j) / (48 - s - j)
        total += temp

    print(f"Probability of playing a straight length {l} (not A, 2):", total)
    for r in range(12 - l):
        straight = ''.join(ranks[r:r+l])
        results.append((straight, total))  # Append valid straight result


# Straight (A, not 2)
for l in range(4, 12):
    total = 0
    for i in generate_valid_tuples(l, 15):
        temp = 1
        for j in i:
            if j == 1 or j == 3:
                temp *= 4
            elif j == 2:
                temp *= 6
        s = sum(i)
        for j in range(s):
            temp *= (16 - j) / (48 - j)
        for j in range(4 * l - s):
            temp *= (32 - j) / (48 - s - j)
        temp *= 1 - (32 - 4 * l + s) / (48 - 4 * l) * (31 - 4 * l + s) / (47 - 4 * l) * (30 - 4 * l + s) / (46 - 4 * l)
        total += temp
    print(f"Probability of playing a straight length {l+1} (A, not 2):", total)
    straight = ''.join(ranks[-l:]) + "A"
    results.append((straight, total))  # Append valid straight result

# Straight (2, not A)
for l in range(4, 12):
    total = 0
    for i in generate_valid_tuples(l, 15):
        temp = 1
        for j in i:
            if j == 1 or j == 3:
                temp *= 4
            elif j == 2:
                temp *= 6
        s = sum(i)
        for j in range(s):
            temp *= (16 - j) / (48 - j)
        for j in range(4 * l - s):
            temp *= (32 - j) / (48 - s - j)
        temp *= (16 - s) / (48 - 4 * l)
        total += temp
    print(f"Probability of playing a straight length {l+1} (2, not A):", total)
    straight = "2" + ''.join(ranks[:l])
    results.append((straight, total))

# Straight (A, 2)
for l in range(3, 12):
    total = 0
    for i in generate_valid_tuples(l, 14):
        temp = 1
        for j in i:
            if j == 1 or j == 3:
                temp *= 4
            elif j == 2:
                temp *= 6
        s = sum(i)
        for j in range(s):
            temp *= (16 - j) / (48 - j)
        for j in range(4 * l - s):
            temp *= (32 - j) / (48 - s - j)
        temp *= (16 - s) / (48 - 4 * l) * (1 - (32 - 4 * l + s) / (47 - 4 * l) * (31 - 4 * l + s) / (46 - 4 * l) * (30 - 4 * l + s) / (45 - 4 * l))
        total += temp
    print(f"Probability of playing a straight length {l+2} (A, 2):", total)
    straight = ''.join(ranks[-l:]) + "A2"
    results.append((straight, total))  # Append valid straight result

for l in range(2, 9):  # Double straight of length 2 to 8
    total = 0
    for i in generate_valid_tuples(l, 16, 2):  # Use backtracking generator
        temp = 1
        for j in i:
            if j == 3:
                temp *= 4
            elif j == 2:
                temp *= 6
        s = sum(i)
        for j in range(s):
            temp *= (16 - j) / (48 - j)
        for j in range(4 * l - s):
            temp *= (32 - j) / (48 - s - j)
        total += temp

    print(f"Probability of playing a double straight length {l} (not A):", total)
    for r in range(12 - l):
        double_straight = ''.join(c * 2 for c in ranks[r:r+l])  # Repeat each character twice
        results.append((double_straight, total))  # Append result

for l in range(1, 8):  # Double straight of length 2 to 8 (A)
    total = 0
    for i in generate_valid_tuples(l, 14, 2):  # Use backtracking generator
        temp = 1
        for j in i:
            if j == 3:
                temp *= 4
            elif j == 2:
                temp *= 6
        s = sum(i)
        for j in range(s):
            temp *= (16 - j) / (48 - j)
        for j in range(4 * l - s):
            temp *= (32 - j) / (48 - s - j)
        temp *= 3 * (16 - s) / (48 - 4 * l) * (15 - s) / (47 - 4 * l) * (32 - 4 * l + s) / (46 - 4 * l) + (16 - s) / (48 - 4 * l) * (15 - s) / (47 - 4 * l) * (14 - s) / (46 - 4 * l)
        total += temp

    print(f"Probability of playing a double straight length {l + 1} (A):", total)
    double = ''.join(c * 2 for c in ranks[-l:]) + "AA"
    results.append((double, total))
    
for l in range(2, 6):  # Triple straight of length 2 to 5 (not A)
    total = 0
    for i in generate_valid_tuples(l, 16, 3):  # Use backtracking generator
        temp = 1
        for j in i:
            if j == 3:
                temp *= 4
        s = sum(i)
        for j in range(s):
            temp *= (16 - j) / (48 - j)
        for j in range(4 * l - s):
            temp *= (32 - j) / (48 - s - j)
        total += temp

    print(f"Probability of playing a triple straight length {l} (not A):", total)
    for r in range(12 - l):
        triple_straight = ''.join(c * 3 for c in ranks[r:r+l])  # Repeat each character thrice
        results.append((triple_straight, total))  # Append result

for l in range(1, 5):  # Triple straight of length 2 to 5 (A)
    total = 0
    for i in generate_valid_tuples(l, 13, 3):  # Use backtracking generator
        temp = 1
        for j in i:
            if j == 3:
                temp *= 4
        s = sum(i)
        for j in range(s):
            temp *= (16 - j) / (48 - j)
        for j in range(4 * l - s):
            temp *= (32 - j) / (48 - s - j)
        temp *= (16 - s) / (48 - 4 * l) * (15 - s) / (47 - 4 * l) * (14 - s) / (46 - 4 * l)
        total += temp

    print(f"Probability of playing a triple straight length {l + 1} (A):", total)
    triple = ''.join(c * 3 for c in ranks[-l:]) + "AAA"
    results.append((triple, total))

# Sort the results by probability
results.sort(key=lambda x: x[1], reverse=True)

# Print the results
for r in results:
    print(r)