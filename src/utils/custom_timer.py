import time


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('Action: %s' % self.name, )
        print('Elapsed: %s' % (time.time() - self.tstart))
