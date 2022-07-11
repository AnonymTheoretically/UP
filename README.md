# Experiments on Undercutting Mining Strategy
This project looks into the profitability of undercutting attack on Bitcoin and Monero, along with a mitigation technique. The analysis is applicable to other PoW-based cryptocurrencies where transactions fees are the major source of incentives. 

The system has three major modules: chains, miners, and unconfirmed transaction sets. Chains module keeps track of states of chains, including height and list of blocks, each of which comprises of size, owner, and fee total. Miners module records information of miners, with mining powers, types, and available strategy sets for every miner. Unconfirmed transaction sets module is also called the memory pool module and tracks namely the unconfirmed transactions of different chains. All the modules evolve in the form of state transitions as new transactions being fed into the system and miners taking action. We allow fees to arrive with transactions instead of at a constant rate. 

The routines in ``computea.py and computeab.py`` respectively compute parameter $a$, the amount of fees to claim in the first block when undercutting and considering defending against another potential undercutter, and $a$ along with $b$, the fee amount to claim in the second block. 
