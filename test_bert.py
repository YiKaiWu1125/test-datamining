import os
import random
import time

def main():
    arr = [1,1,1,2,3,2,1,1,1,1,1]
    while True:
        print("Current time:", time.strftime("%H:%M:%S", time.localtime()))
        
        epochs = arr[random.randint(0, len(arr) - 1)]
        
        weight_decay = random.uniform(0.01, 0.3) # next 0.2-.0.5
        weight_decay = round(weight_decay, 2)

        batch_size = random.randint(2, 35) 
        
        print(f"num_epochs:{epochs} & batch_size:{batch_size} & weight_decay:{weight_decay}")
        os.system(str("python main-bert2.py -epochs " + str(epochs) + " -batch_size "+ str(batch_size) +" -weight_decay "+str(weight_decay)))

if __name__ == "__main__":
    main()