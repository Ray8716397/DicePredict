import random
import multiprocessing


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


def predict_proc(pidx, lock):
    random.seed(random.seed(random.randint(1, 100)))
    data = [9, 12, 16, 10, 13, 12, 8, 12, 8, 11, 15, 15, 10, 9, 8, 11, 16, 12, 13, 11, 6, 13, 9, 11, 10, 9, 8, 12, 17,
            16, 11, 10, 6, 6, 10, 10]
    length = 35
    if len(data) > length:
        data = data[len(data) - length:]
    seed = 1
    max_epoch = 10
    count_poch = 0
    while count_poch < max_epoch:
        count_odd = 0
        count_small = 0
        for i, d in enumerate(data[:-1]):
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

    lock.acquire()
    print(f"---------------------------------------------------------------------------------------------------")
    print(f"proc{pidx} res: \033[1;31m {predict_roll([x for x in data])} \033[0m")
    print(f"odd_acc: {odd_acc}, odd%: {sum([1 if i % 2 == 1 else 0 for i in data]) / len(data)}")
    print(f"small_acc: {small_acc}, small%: {sum([1 if i < 11 else 0 for i in data]) / len(data)}")
    lock.release()


with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as proc_pool:
    manager = multiprocessing.Manager()
    plock = manager.Lock()
    for xx in range(4):
        proc_pool.apply_async(predict_proc, (xx, plock))

    else:
        proc_pool.close()
        proc_pool.join()
