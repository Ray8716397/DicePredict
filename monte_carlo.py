import random


def roll_dice():
    return random.randint(1, 6)


def predict_roll(history, num_trials=10000):
    counts = [0] * 16
    for i in range(num_trials):
        roll1 = roll_dice()
        roll2 = roll_dice()
        roll3 = roll_dice()
        roll = roll1 + roll2 + roll3
        if roll in history:
            counts[roll - 4] += 1
    total_counts = sum(counts)
    probabilities = [count / total_counts for count in counts]
    predicted_roll = random.choices(range(4, 20), probabilities)[0]
    return predicted_roll

data = [[3, 1, 1], [6, 5, 2], [4, 4, 4], [6, 3, 1], [6, 2, 1], [5, 3, 2], [5, 4, 3], [6, 4, 3], [6, 4, 2], [5, 3, 2],
        [5, 4, 1], [5, 3, 1], [3, 2, 1], [6, 5, 4], [4, 4, 2], [6, 5, 4], [3, 3, 1], [5, 2, 1], [6, 6, 3], [6, 4, 3],
        [4, 3, 3], [6, 4, 4], [2, 2, 1], [6, 5, 1], [4, 2, 2], [5, 2, 1], [2, 2, 1], [6, 3, 1], [4, 3, 1], [6, 2, 1],
        [4, 4, 1], [5, 1, 1], [6, 5, 3], [5, 2, 1], [6, 6, 4], [4, 3, 3], [3, 2, 2], [5, 5, 2], [6, 6, 2], [6, 3, 2],
        [3, 2, 2], [6, 5, 2], [4, 3, 2], [2, 1, 1], [5, 3, 2], [5, 3, 2], [3, 2, 2], [2, 2, 1], [5, 3, 2], [6, 5, 4],
        [6, 4, 4], [6, 3, 1], [4, 3, 2], [6, 4, 2], [6, 4, 4], [4, 4, 3], [5, 5, 5], [4, 3, 1], [2, 2, 2], [5, 4, 3],
        [6, 1, 1], [5, 2, 2], [2, 1, 1], [4, 2, 2], [6, 5, 1], [5, 4, 1], [6, 6, 1]]

print(predict_roll([sum(x) for x in data]))
count_odd = 0
count_small = 0
for i in range(len(data)-1):
    res = predict_roll([sum(x) for x in data[:i+1]])
    r_odd = res % 2 == 1
    r_small = res < 11
    t_odd = sum(data[i+1]) % 2 == 1
    t_small = sum(data[i+1]) < 11
    if r_odd == t_odd:
        count_odd += 1
    if r_small == t_small:
        count_small += 1
odd_acc = count_odd / len(data)
small_acc = count_small / len(data)

print(odd_acc)
print(small_acc)
