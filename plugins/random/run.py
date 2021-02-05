import sys
import random


def test(charge):
    rand = random.randint(0, 10)
    print(charge)
    sys.stdout.write(str(rand)+"\n")
    return rand, 3 * rand


def run(snap):
    charge = snap.charge
    print(snap.plugin_args['michaels_test_message'])
    return test(charge)
