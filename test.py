import os
import random

def logarithmic_distribution(low, high, bias=0.1):
    return low + (high - low) * random.uniform(0, 1) ** bias

def main():
    while True:
        #num_epochs = random.randint(2, 20)
        num_epochs = int(logarithmic_distribution(2, 20))
        dropout_rate = random.uniform(0.2, 0.4)
        dropout_rate = round(dropout_rate, 2)
        os.system(str("python main.py -epochs " + str(num_epochs) + " -dropout "+ str(dropout_rate) ))

if __name__ == "__main__":
    main()