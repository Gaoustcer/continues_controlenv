import multiprocessing
from time import time
arr = []
N = 1024 * 1024 * 1024
def _add_element():
    for i in range(N):
        arr.append(i**0.01)

def para():
    processlist = []
    N = 5
    
    for _ in range(N):
        processlist.append(multiprocessing.Process(target=_add_element))
    start = time()
    for i in range(N):
        processlist[i].start()
    for i in range(N):
        processlist[i].join()

    end = time()
    print('time spend is',end - start,len(arr))

def cpunopar():
    start = time()
    _add_element()
    end = time()
    print("Time is",end - start,len(arr))

if __name__ == "__main__":
    cpunopar()