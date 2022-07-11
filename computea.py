# Compute the fees to claim in the first block on a fork
# inputs: mining power distributions; fees in bandwidth set
# output: the maximum claimable fees for the strongest undercutter (parameterized with "a"); maximum claimable fees for the second strongest undercutter (parameterized with "b")
# Author: Tiantian Gong <tg@purdue.edu>
# Setting up params
betah = 0.0 # honest mining power
betau = 0.176 # strongest rational miner
betau2 = 0.176 # second strongest rational miner
betar = 1 - betah - betau - betau2 #remaining rational miners
assert(betah>=0)
assert(betau>0)
assert(betau2>=0)
assert(betar>=0)
gamma = 0.9 # Fees in the next bandwidth set (measured relative to the current bandwidth set) 
gamma2 = 0.8 # Fees in the next bandwidth set (measured relative to the current bandwidth set) when it's the second strongest rational miner's time to move

a = 1.0 # Parameter for the strongest rational miner
b = 1.0 # Parameter for the second strongest rational miner
Ta = (1 + gamma)*(1 - betau2)/(1 + (betar + betau) * b)
Tb = (1 + gamma2)*(1 - betau)/(1 + (betar + betau2) * a)
ctr = 100 # In case the program does not terminate soon
while True:
    ctr -= 1
    if ctr == 0:
        break
    if a < Ta:
        if b < Tb:
            print(a)
            print(b)
            break
        else:
            b -= 0.01
            Ta = (1 + gamma)*(1 - betau2)/(1 + (betar + betau) * b) # (betar + betau) is \beta_{r_2} in Ineqaulities (3)
            a = Ta - 0.01
            Tb = (1 + gamma2)*(1 - betau)/(1 + (betar + betau2) * a)
    else:
        b -= 0.01
        Ta = (1 + gamma)*(1 - betau2)/(1 + (betar + betau) * b)
        a = Ta - 0.01
        Tb = (1 + gamma2)*(1 - betau)/(1 + (betar + betau2) * a)
    
    
            
