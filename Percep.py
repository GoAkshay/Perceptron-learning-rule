import numpy as np

class bcolors:
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def get_input_data():
    """Function to gather user input for learning constant, dataset, and desired outputs."""
    c = float(input("\nEnter learning constant C: "))
    n = int(input("Enter number of dataset: "))
    
    xlist = []
    wlist = [np.array(list(map(float, input("Enter values for W1: ").split()))).reshape(-1, 1)]
    
    for i in range(n):
        x = np.array(list(map(float, input(f"Enter values for X{i+1}: ").split()))).reshape(-1, 1)
        xlist.append(x)
    
    d = np.array(list(map(int, input("Enter desired outputs d: ").split())))
    
    return c, n, xlist, wlist, d

def perceptron_learning(c, n, xlist, wlist, d):
    """Function to implement the Perceptron learning algorithm."""
    print(f"\n{bcolors.FAIL}{bcolors.UNDERLINE}Solution:{bcolors.ENDC}")
    
    for i in range(n):
        # Calculate net input
        net = np.dot(wlist[-1].T, xlist[i])
        op = 1 if net > 0 else -1 if net < 0 else 0

        # Check if output matches the desired output
        if op == d[i]:
            print(f"{bcolors.FAIL}{bcolors.BOLD}No adjustment needed! for W{i+2}{bcolors.ENDC}\n")
            wlist.append(wlist[-1])  # No change to weights
        else:
            # Adjust weights
            r = c * (d[i] - op)
            temp = np.dot(r, xlist[i].T).reshape(-1, 1)
            wlist.append(wlist[-1] + temp)
            print(f"{bcolors.FAIL}{bcolors.BOLD}W{i+2} is{bcolors.ENDC}")
            print(wlist[-1])
            print("\n")
    
    return wlist

def main():
    c, n, xlist, wlist, d = get_input_data()
    final_weights = perceptron_learning(c, n, xlist, wlist, d)
    print(f"Final Weights: {final_weights[-1]}")

if __name__ == "__main__":
    main()
