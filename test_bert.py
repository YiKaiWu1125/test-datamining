import os
import random
import time

def main():
    num_epochs = 1
    while True:
        print("Current time:", time.strftime("%H:%M:%S", time.localtime()))
        os.system(str("python main-bert2.py -epochs " + str(num_epochs) ))
        num_epochs += 1


if __name__ == "__main__":
    main()