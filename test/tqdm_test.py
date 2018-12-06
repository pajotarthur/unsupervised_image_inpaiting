from tqdm import trange
from random import random, randint
from time import sleep

# with trange(100) as t:
#     for i in t:
#         # Description will be displayed on the left
#         t.set_description('GEN %i' % i)
#         # Postfix will be displayed on the right,
#         # formatted automatically based on argument's datatype
#         t.set_postfix(loss=random(), gen=randint(1,999), str='h',
#                       lst=[1, 2])
#         sleep(0.1)
from tqdm import tqdm
with tqdm(total=10, bar_format="{desc}: {percentage:3.0f}%|{bar}|{n_fmt}/{total_fmt} {rate_fmt}{postfix}]") as t:
    for i in range(10):
        sleep(0.1)
        # t.postfix[1]["value"] = i / 2
        # {l_bar}{bar}{r_bar}
        t.update()