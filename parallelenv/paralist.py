import multiprocessing as mp 

from time import time
# arr = []
# N = 1024 * 1024 * 60
def _add_element(arr,index,dict):
    # global arr
    # global N
    N = 1024 ** 2 * 60
    dict[index] = []
    for i in range(N):
        dict[index].append(i**0.01)
    print("call finished",len(arr))

def para():
    processlist = []
    N = 5
    arr = []
    dic = mp.Manager().dict()
    for i in range(N):
        processlist.append(mp.Process(target=_add_element,args=(arr,i,dic)))
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
    # cpunopar()
    para()