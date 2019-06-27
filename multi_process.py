#rom multiprocessing import Queue, Process
import time
from Queue import Queue
from threading import Thread

"""
Dead Lock 
def writter(q):
    q.put(['x'] * 30000)
    print 'putting in queue finished'

def reader():
    pass
    
def main():
    queue = Queue()
    p = Process(target = writter, args = (queue, ))
    p.start()
    time.sleep(2)
    p.join()
    print 'Finished'
"""


"""
multi-process
def calc1(q, df):
    for i in range(5):
        print 'calc1 loop = {}'.format(i)
        df[0] +=1
        time.sleep(5)
    q.put(df)

def calc2(q,df):
    for i in range(5):
        print 'calc2 loop = {}'.format(i)
        df[0] -=1
    q.put(df)


def main():
    df = [0]
    queue = Queue()
    result = []
    thread1 = Thread(target = calc1, args = (queue, df))
    thread2 = Thread(target = calc2, args = (queue, df))
    thread1.start()
    thread2.start()

    for i in range(2):
        result.append(queue.get())

    thread1.join()
    thread2.join()
    print result
"""






if __name__ == '__main__':
    main()