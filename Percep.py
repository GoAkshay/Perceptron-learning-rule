import numpy as mm
class bcolors:
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

c=float(input("\n Enter learning constant C: "))
n=int(input(" Enter number of dataset: "))
xlist=list()
wlist=list()
wlist.append(mm.array(list(map(float, input(" Enter values for W1: ").split()))))
wlist[0]=wlist[0].reshape(len(wlist[0]),1)

for i in range(n):
    xlist.append(mm.array(list(map(float, input(f" Enter values for X{i+1}: ").split()))))
    xlist[i]=xlist[i].reshape(len(xlist[i]),1)
d=list()
d.append(list(map(int,input(" Enter desired outputs d: ").split())))
print("\n")
print(f"{bcolors.FAIL}{bcolors.UNDERLINE}Solution:\n{bcolors.ENDC}")

for i in range(0,len(xlist)):
    wt = wlist[i].transpose()
    net = mm.dot(wt, xlist[i])
    if(net>0):
        op=1
    elif(net<0):
            op=-1
    else:
        op=0
    if(op==d[0][i]):
        print(f"{bcolors.FAIL}{bcolors.BOLD}No adjustment needed! for W"+str(i+2), f"{bcolors.ENDC}")
        print("\n")
        wlist.append(wlist[i])
    else:
        r=c*(d[0][i]-op)
        temp=mm.dot(r,xlist[i])
        wlist.append(mm.add(wlist[i],temp))
        print(f"{bcolors.FAIL}{bcolors.BOLD}W" + str(i + 2), f"is{bcolors.ENDC}")
        print(wlist[i + 1])
        print("\n")
