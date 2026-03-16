import numpy as np 
import subprocess
import matplotlib.pyplot as plt
'''plotter for the c++ executabl'''
if __name__ == '__main__':
    result = subprocess.run(["./appman.exe"],
                            capture_output = True,
                            text = True,
                            timeout=None)
    # you can do it with valid json too. 
    lines = result.stdout.strip().splitlines()
    for line in lines:
        print(line)

    data = [list(map(np.float16, li.split(','))) for li in lines[2::]]
    data = np.array(data)
    plt.figure()
    plt.subplot(2,1,1)
    plt.scatter(data[:,0], data[:,2])
    plt.subplot(2,1,2)
    plt.scatter(data[:,1], data[:,2])
    plt.show()