import os
import random

def main():
    while True:
        num_epochs = random.randint(20, 50)
        dropout_rate = random.uniform(0.6, 0.6)
        dropout_rate = round(dropout_rate, 2)
        os.system(str("python main.py -epochs " + str(num_epochs) + " -dropout "+ str(dropout_rate) ))

if __name__ == "__main__":
    main()