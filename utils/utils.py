import os
import codecs
import argparse
import time
import sys
import json
import subprocess
import numpy as np
import random
import string


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in ['off', 'false', '0']:
        return False
    if s.lower() in ['on', 'true', '1']:
        return True
    raise argparse.ArgumentTypeError("invalid value for a boolean flag (0 or 1)")


def getcurrentgithash():
    git_repo = subprocess.check_output(["git", "config", "--get", "remote.origin.url"]).strip()
    git_repo = git_repo.split(b"/")[-1]
    #git_repo = b"None"
    label = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip()
    #label = label[:7]
    return git_repo.decode('utf-8') + ":" + label.decode('utf-8')


def gettimecommand(myargs):
    TFORMAT = "%Y%m%d %H:%M:%S"
    mytstruct = time.gmtime()
    newsecs = time.strftime(TFORMAT, mytstruct)
    myhist = ''
    myhist += os.uname()[1] + ' '
    myhist += str(os.getpid()) + ' '
    myhist += newsecs + ' python '
    myhist += " ".join(myargs) + '\n'
    return myhist


MYGLOBALHASH = getcurrentgithash()
MYGLOBALCMD = gettimecommand(sys.argv)


def parse_args(myargs, desc, types):
    if isinstance(myargs, dict):
        return myargs
    print_args = False
    if len(myargs) == 0 and len(desc) > 0 and len(types) > 0:
        myargs = ['1'] * len(desc)
        print_args = True
    assert len(myargs) == len(desc) == len(types), 'Argument parse length mismatch: ' + str(len(myargs)) + "/" + str(len(desc)) + "/" + str(len(types)) + "."
    result = {}
    for d, x, t in zip(desc, myargs, types):
        if t == 's':
            result[d] = str(x)
        elif t == 'i':
            assert x.isdigit(), 'Argument cannot be converted to integer.'
            result[d] = int(x)
        elif t == 'f':
            assert x.replace('.', '').isdigit(), 'Argument cannot be converted to float.'
            result[d] = float(x)
        elif t == 'b':
            assert x.isdigit(), 'Argument cannot be converted to binary.'
            result[d] = bool(int(x))
        elif t == 'e':
            result[d] = x
        else:
            assert 0, 'Unknown variable type.'
    if print_args:
        print(result)
        return None
    else:
        return result


class bag(object):
    def __init__(self):
        pass


def mkhist(path):
    path = "/".join(path.split("/")[:-1])
    hist = open(os.path.join(path, '.pdhist'), 'a')
    hist.write(MYGLOBALHASH + " ")
    hist.write(MYGLOBALCMD + " ")
    hist.write(path + "\n")
    hist.close()


def store(path):
    assert os.path.exists(path) == 0, path + " exists already."
    print("created:", path)
    mkhist(path)
    return open(path, 'w', encoding='utf-8')


def bstore(path):
    assert os.path.exists(path) == 0, path + " exists already."
    print("created:", path)
    mkhist(path)
    return open(path, 'wb')


def read(path):
    print("opened:", path)
    return open(path, 'r', encoding='utf-8')


def bread(path):
    print("opened:", path)
    return open(path, 'rb')


def costore(path):
    assert os.path.exists(path) == 0, path + " exists already."
    print("created:", path)
    mkhist(path)
    return codecs.open(path, 'w', encoding='utf-8', errors='xmlcharrefreplace')


def coread(path):
    print("opened:", path)
    return codecs.open(path, 'r', encoding='utf-8', errors='xmlcharrefreplace')


def mkdir(path):
    assert os.path.exists(path) == 0, path + " exists already."
    print("created directory:", path)
    os.mkdir(path)


def ls(path):
    return os.listdir(path)


class status(object):
    def __init__(self, interval, total=''):
        self.count = 0
        self.interval = interval
        self.old_time = time.time()
        self.total = total

    def rep(self):
        self.count += 1
        if self.count % self.interval == 0:
            print("iteration: " + str(self.count) + " / " + str(self.total) + " (" + str(round(time.time() - self.old_time, 2)) + " seconds)")
            self.old_time = time.time()


def byte2str(a):
    if not a.isdigit():
        print("No conversion needed", a)
        return a
    ba = bytearray()
    ba.extend([int(a[i * 3:(i + 1) * 3]) for i in range(len(a) // 3)])
    #return ''.join([chr(int(a[i * 3:(i + 1) * 3])) for i in range(len(a) // 3)])
    # todo perhaps better to return the bytestr?
    return ba.decode('utf-8')


def str2byte(a):
    if a.isdigit():
        # todo this is a bug as numbers are not converted correctly!
        print("No conversion needed", a)
        return a
    if not isinstance(a, bytes):
        a = a.encode('utf-8')
    #return ''.join([str(ord(x)).zfill(3) for x in a])
    return ''.join([str(x).zfill(3) for x in a])


def getnesteddefaultdict(n_levels, lambda_):
    raise NotImplementedError


def invdict(dict_):
    if len(set(dict_.values())) != len(dict_.values()):
        raise ValueError("Values in dict are not unique, cannot reverse.")
    return dict([(v, k) for k, v in dict_.items()])


def describe(list_):
    list_ = np.array(list_)
    print("{0:>12} | {1:>12} | {2:>12} | {3:>12} | {4:>12} | {5:>12}".format("Mean", "Median", "Std", "Max", "Min", "N"))
    print("{0:12.2f} | {1:12.2f} | {2:12.2f} | {3:12.2f} | {4:12.2f} | {5:12.0f}".format(np.mean(list_), np.percentile(list_, 50), np.std(list_), np.max(list_), np.min(list_), len(list_)))


def load_config(config):
    with open(config) as f:
        data = json.load(f)
    return data


def chunks(list_, chunk_size):
    for i in range(0, len(list_), chunk_size):
        yield list_[i:i+chunk_size]


def get_randomid(length=5):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))





