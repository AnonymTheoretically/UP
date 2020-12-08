# UndercuttingProject
This project looks into the profitability of undercutting attack on Bitcoin and Monero, along with a possible mitigation technique. The analysis is applicable to other PoW-based cryptocurrencies. PoS and others necessaitate updates of "mining powers" to be interpreted as "stakes" and others.

Conceptually, the model has three major modules: chains, miners, and unconfirmed transaction sets. Chains module keeps track of states of chains, including height and list of blocks, each of which comprises of size, owner, and fee total. Miners module records information of miners, with mining powers, types, and available strategy sets for every miner. Unconfirmed transaction sets module is also called the memory pool module and tracks namely the unconfirmed transactions of different chains. All the modules evolve in the form of state transitions as new transactions being fed into the system and miners taking action. We allow fees to arrive with transactions instead of at a constant rate. 

You can find the write-up for the findings of this project here: https://arxiv.org/abs/2007.11480

Please feel free to contact the authors if you should have any questions about the model or the experiment codes. 