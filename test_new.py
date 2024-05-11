import os
import random
import time

def main():  
    epochs_arr =       [  1,    20,   20,    20,    20,    1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1]
    weight_decay_arr = [0.3,  0.25,  0.3,  0.35,   0.4, 0.31,  0.33,  0.35,  0.38,   0.4,  0.42,  0.45,  0.48,   0.5,  0.52,  0.55,  0.58,   0.6,  0.62,  0.65,  0.68,   0.7,  0.72,  0.75,  0.78,   0.8,  0.82,  0.85,  0.88]
    batch_size_arr =   [  4,     4,    4,     4,     4,    8,     8,     8,     8,     8,     8,     8,     8,     8,     8,     8,     8,     8,     8,     8,     8,     8,     8,     8,     8,     8,     8,     8,     8]
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