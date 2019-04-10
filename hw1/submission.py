import numpy as np
def submission(file,out):
    load = np.load(file)
    fle = open(out,'w')
    fle.write('id,label'+'\n')
    for i in range(np.shape(load)[0]):
        fle.write(str(i) + ','+str(load[i])+'\n')
    fle.close()

if __name__ == '__main__':
    submission('label_test.npy', 'submission.csv')
    