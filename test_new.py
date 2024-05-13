import os
import random
import time

def main():  
    epochs_arr =       [  3,    1 ,   1]
    weight_decay_arr = [0.4,  0.4 , 0.3]
    batch_size_arr =   [  8,    8 ,   8]
    index = 0
    while True:
        print("Current time:", time.strftime("%H:%M:%S", time.localtime()))
        
        epochs = epochs_arr[index] #arr[random.randint(0, len(arr) - 1)]
        
        #weight_decay = random.uniform(0.01, 0.3) # next 0.2-.0.5
        weight_decay = weight_decay_arr[index] #round(weight_decay, 2)

        batch_size = batch_size_arr[index] #random.randint(2, 35) 
        
        index += 1
        
        print(f"num_epochs:{epochs} & batch_size:{batch_size} & weight_decay:{weight_decay}")
        os.system(str("python new.py -epochs " + str(epochs) + " -batch_size "+ str(batch_size) +" -weight_decay "+str(weight_decay)))

if __name__ == "__main__":
    main()