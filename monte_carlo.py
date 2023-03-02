import random


def roll_dice():
    return random.randint(1, 6)


def predict_roll(history, num_trials=1000):
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


for xx in range(5):
    print("---------------------------------------------------------------------------------------------------")
    random.seed(random.seed(random.randint(1, 100)))
    data = [18, 7, 9, 17, 16, 13, 11, 9, 9, 12, 11, 4, 14, 8, 8, 8, 9, 10, 10, 18, 6, 10, 14, 12, 7, 15, 13, 11, 12, 4,
            7,
            14, 14, 13, 11, 15, 11, 8, 11, 11, 11, 12, 15, 6, 16, 8, 11, 10, 12, 13, 13, 11, 12, 10, 11, 9, 8, 11, 12,
            15,
            12, 12, 11, 8, 15, 13, 13, 16, 9, 8, 9, 8, 12, 5, 11, 8, 9, 12, 8, 12, 9, 5, 13, 10, 8, 12, 12, 8, 14, 9, 9, 5, 16]
    length = 35
    data = data[len(data) - length:]
    seed = 1
    max_epoch = 10
    count_poch = 0
    while count_poch < max_epoch:
        count_odd = 0
        count_small = 0
        for i in range(length - 1):
            res = predict_roll([x for x in data[:i + 1]])
            r_odd = res % 2 == 1
            r_small = res < 11
            t_odd = data[i + 1] % 2 == 1
            t_small = data[i + 1] < 11
            if r_odd == t_odd:
                count_odd += 1
            if r_small == t_small:
                count_small += 1
        odd_acc = count_odd / length
        small_acc = count_small / length

        if small_acc > 0.55 and odd_acc > 0.55:
            break
        else:
            seed += 1
            count_poch += 1

    print(predict_roll([x for x in data]))
    print(f"odd_acc: {odd_acc}, odd%: {sum([1 if i % 2 == 1 else 0 for i in data]) / len(data)}")
    print(f"small_acc: {small_acc}, small%: {sum([1 if i < 11 else 0 for i in data]) / len(data)}")
