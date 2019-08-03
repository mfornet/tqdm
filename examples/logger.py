from tqdm import tqdm, TqdmLogger as logger

s = 0

for i in tqdm(range(10**7)):
    s += i
    num = i % (2 * 10**6)

    if num >= 10**6:
        logger.clear()
    else:
        logger.log(str(i))

print(s)
