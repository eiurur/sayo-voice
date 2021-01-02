import time


class StopWatch:
    board = {}

    @staticmethod
    def start(label):
        StopWatch.board[label] = time.perf_counter()

    @staticmethod
    def end(label):
        t = time.perf_counter() - StopWatch.board[label]
        print('Elapsed time -> {} :{:.7}'.format(label, t))
