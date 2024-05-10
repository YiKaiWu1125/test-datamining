import os
import random
import time

def main():  
    #arr = [1,1,1,2,3,2,1,1,1,1,1]
    epochs_arr =       [    2,    1,   1,   1,    1,   1,    1,   1,    1 ]
    weight_decay_arr = [  0.2,  0.2, 0.3, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45 ]
    batch_size_arr =   [   32,   32,  32,  64,   25,  30,   30,  30,  128 ]
    index = 0
    while True:
        print("Current time:", time.strftime("%H:%M:%S", time.localtime()))
        
        epochs = epochs_arr[index] #arr[random.randint(0, len(arr) - 1)]
        
        #weight_decay = random.uniform(0.01, 0.3) # next 0.2-.0.5
        weight_decay = weight_decay_arr[index] #round(weight_decay, 2)

        batch_size = batch_size_arr[index] #random.randint(2, 35) 
        
        index += 1
        
        print(f"num_epochs:{epochs} & batch_size:{batch_size} & weight_decay:{weight_decay}")
        os.system(str("python main-bert2.py -epochs " + str(epochs) + " -batch_size "+ str(batch_size) +" -weight_decay "+str(weight_decay)))

if __name__ == "__main__":
    main()