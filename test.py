import os
import random

def main():
    while True:
        num_epochs = random.randint(2, 20)
        dropout_rate = random.uniform(0.2, 0.)
        dropout_rate = round(dropout_rate, 2)
        os.system(str("python main.py -epochs " + str(num_epochs) + " -dropout "+ str(dropout_rate) ))

if __name__ == "__main__":
    main()