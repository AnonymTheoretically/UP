# Compute the fees to claim in the first two blocks on a fork
# inputs: mining power distributions; fees in bandwidth set
# output: the maximum claimable fees for the strongest undercutter (parameterized with "a,b"); maximum claimable fees for the second strongest undercutter (parameterized with "c,d")
# Author: Tiantian Gong <tg@purdue.edu>
# Setting up params

betah = 0.2 # honest mining power
betau = 0.25 # strongest rational miner
betau2 = 0.2 # second strongest rational miner
betar = 1 - betah - betau - betau2 #remaining rational miners
assert(betah>=0)
assert(betau>0)
assert(betau2>=0)
assert(betar>=0)
gamma = 0.5 # Fees in the next bandwidth set (measured relative to the current bandwidth set) 
gamma2 = 0.4 # Fees in the next bandwidth set (measured relative to the current bandwidth set) when it's the second strongest rational miner's time to move
gamma3 = 0.4 # Fees in the thrid bandwidth set (measured relative to the next bandwidth set) 
gamma4 = 0.4 # Fees in the third bandwidth set (measured relative to the next bandwidth set) when it's the second strongest rational miner's time to move

a = 1.0 # Parameter a for the strongest rational miner
b = 1.0 # Parameter b for the strongest rational miner
c = 1.0 # Parameter a for the second strongest rational miner
d = 1.0 # Parameter b for the second strongest rational miner
# Compute T, T'
t1 = ((betau2 + betar + betau) * a + betau2 * b + betau2 *(betau2 + betar + betau)*(1 - a - b))/(2 * (1 - betau2 * (betau2 + betar + betau))) #T
t2 = ((betau2 + betar + betau) * a + betau * b + betau *(betau2 + betar + betau)*(1 - a - b))/(2 * (1 - betau * (betau2 + betar + betau))) #T'
if betah > betau2:
    t1 = min(t1, ((betau2 + betar + betau) * (1 - a) + betah * b) / (2*(betah - betau2)) )
if betah > betau:
    t2 = min(t2, ((betau2 + betar + betau) * (1 - a) + betah * b) / (2*(betah - betau)))

# Compute attack condition
Ta = (1 + gamma)/(1 + t1)
Tb = (1 + 2 * gamma2 - a)/(1 + t1)
Tc = (1 + gamma3)/(1 + t2)
Td = (1 + 2 * gamma4 - a)/(1 + t2)

ctr = 100 # In case the program does not terminate soon
while True:
    ctr -= 1
    if ctr == 0:
        break
    if a < Ta and b < Tb:
        if c < Tc and d < Td:
            print(a)
            print(b)
            print(c)
            print(d)
            break
        else:
            if d >= Td:
                d -= 0.01
            else:
                c -= 0.01
            t1 = ((betau2 + betar + betau) * a + betau2 * b + betau2 *(betau2 + betar + betau)*(1 - a - b))/(2 * (1 - betau2 * (betau2 + betar + betau))) #T
            t2 = ((betau2 + betar + betau) * a + betau * b + betau *(betau2 + betar + betau)*(1 - a - b))/(2 * (1 - betau * (betau2 + betar + betau))) #T'
            if betah > betau2:
                t1 = min(t1, ((betau2 + betar + betau) * (1 - a) + betah * b) / (2*(betah - betau2)) )
            if betah > betau:
                t2 = min(t2, ((betau2 + betar + betau) * (1 - a) + betah * b) / (2*(betah - betau)))
            # Compute attack condition
            Ta = (1 + gamma)/(1 + t1)
            Tb = (1 + 2 * gamma2 - a)/(1 + t1)
            Tc = (1 + gamma3)/(1 + t2)
            Td = (1 + 2 * gamma4 - a)/(1 + t2)
            a = min( Ta - 0.01,1)
            b = min( Tb - 0.01,1)
    else:
        if d >= Td:
            d -= 0.01
        else:
            c -= 0.01
        t1 = ((betau2 + betar + betau) * a + betau2 * b + betau2 *(betau2 + betar + betau)*(1 - a - b))/(2 * (1 - betau2 * (betau2 + betar + betau))) #T
        t2 = ((betau2 + betar + betau) * a + betau * b + betau *(betau2 + betar + betau)*(1 - a - b))/(2 * (1 - betau * (betau2 + betar + betau))) #T'
        if betah > betau2:
            t1 = min(t1, ((betau2 + betar + betau) * (1 - a) + betah * b) / (2*(betah - betau2)) )
        if betah > betau:
            t2 = min(t2, ((betau2 + betar + betau) * (1 - a) + betah * b) / (2*(betah - betau)))
        # Compute attack condition
        Ta = (1 + gamma)/(1 + t1)
        Tb = (1 + 2 * gamma2 - a)/(1 + t1)
        Tc = (1 + gamma3)/(1 + t2)
        Td = (1 + 2 * gamma4 - a)/(1 + t2)
        a = min( Ta - 0.01,1)
        b = min( Tb - 0.01,1)