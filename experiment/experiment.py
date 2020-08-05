import time
from scipy.stats import geom
import math
import numpy as np
import scipy.integrate as integrate
import json
from chain import Chain
from block import Block
from miner import Miner, NormalMiner, UndercutMiner, HonestMiner
from tx import Tx
import random
from multiprocessing import Pool
import time
import itertools
import gc



begining_timestamp = 1589540396
ending_timestamp = 1592265346
tx_file_path = "data/summary_txs_2020_June"

global all_txs
all_txs = {}
def load_txs(tx_file):
    for id, l in enumerate(open(tx_file)):
        print(id)
        tx = json.loads(l)
        try:
            all_txs[tx['timestamp']].append(tx)
        except:
            all_txs[tx['timestamp']] = [tx]


class Simulation():
    def __init__(self, genesis_block, miners, tx_file, start_timestamp, end_timestamp, t, i, cutoff, avg_block_time = 600):
        self.experiment_end_flag = False
        self.undercutter_cutoff = cutoff
        self.num_forks = 0
        self.num_failed_forks = 0
        self.num_failed_forks_cutoff = 0
        self.conf_blocks = 6
        self.tx_file = tx_file
        self.genesis_block = genesis_block
        self.miners = miners
        self.start_time = start_timestamp
        self.current_time = start_timestamp
        self.end_time = end_timestamp
        self.avg_block_time = avg_block_time
        self.avg_block_time_minutes = self.avg_block_time/60
        main_chain = Chain(miners, None ,genesis_block)
        self.chains = [main_chain]
        self.chains_headerblock = {main_chain: (self.current_time + geom.rvs(p=main_chain.get_mining_power_sum()/self.avg_block_time), main_chain.get_mining_power_sum())}
        for miner in self.miners:
            miner.set_current_block(self.genesis_block)
        self.update_mempool(self.genesis_block, self.current_time-100, self.current_time)



    def run(self):
        num_blocks = 0
        while not self.experiment_end_flag:
            num_blocks += 1

            extending_chain = sorted(self.chains_headerblock.keys(), key=lambda x: self.chains_headerblock[x])[0]  # the chain that is going to get extended next

            self.current_time = self.chains_headerblock[extending_chain][0] # update current time to the time of the block creation

            m = np.random.choice(extending_chain.miners, p=extending_chain.get_mining_weights())  # miner that mines the block

            m.set_current_block(extending_chain.get_block(-1)) # set the parent node for proper linking

            next_block = m.publish_next_block(self.current_time)  # get the block to be published by the miner

            self.update_chain(extending_chain, next_block, curr_time=self.current_time) # update the current chain that is extended

            self.remove_forked_chains(extending_chain) # remove any forks that have already lost.

            self.fork(extending_chain)  # undercutter forking the main chain

            self.update_miners_chain(extending_chain) # updating all the remaining miners' working chain

            for chain, (t_old, mining_power_old) in self.chains_headerblock.items():   # update the next block times according to the new miners that have joined
                if chain.get_mining_power_sum() == 0: # no miner working on the chain
                    t_new = self.end_time*2  # equivalent to infinity in the experiment
                else:
                    t_new = ((t_old - self.current_time) * (mining_power_old/chain.get_mining_power_sum()) + self.current_time)
                self.chains_headerblock[chain] = (t_new, chain.mining_power_sum)

            self.update_mempool(next_block, prev_time=extending_chain.chain[-2].timestamp, curr_time=self.current_time)  #update the mempools after the blocks generation

        return next_block, self.num_forks, self.num_failed_forks, self.num_failed_forks_cutoff, num_blocks



    # ===========================================================================================================
    #   functions used in the run method above.
    # ===========================================================================================================

    def fork(self, extended_chain):
        for miner in extended_chain.miners:
            if isinstance(miner, UndercutMiner) and not miner.infork and extended_chain.get_block(-1).miner != miner: # we fork only if the miner is of type undercutter and is not actively in a fork
                forked_chain = Chain(miners=[miner], main_chain=extended_chain, origin_block=extended_chain.get_block(-1).parent_block)
                self.chains.append(forked_chain)
                self.chains_headerblock[forked_chain] =  (self.current_time + geom.rvs(p=forked_chain.get_mining_power_sum()/self.avg_block_time), forked_chain.get_mining_power_sum())
                extended_chain.miners.remove(miner)  # remove the miner from the miner list of prev chain
                extended_chain.update_mining_power()
                miner.infork = True
                self.num_forks += 1


    def update_mempool(self, new_block, prev_time, curr_time):
        txs = self.get_mempool_tx(prev_time, curr_time)  # get new txs for mempool
        new_block.remove_txs_mempool(new_block.txs)
        new_block.add_txs_mempool(txs)  # add new txs to the last newly created block mempool


    def update_chain(self, chain, block, curr_time):
        chain.get_block(-1).next_block = block
        chain.add_block(block) #add the block to the chain
        self.chains_headerblock[chain] =  (curr_time + geom.rvs(p=chain.get_mining_power_sum()/self.avg_block_time), chain.get_mining_power_sum())


    def remove_forked_chains(self, extended_chain):
        c1 = extended_chain
        for c2 in self.chains:  # iterate through all the chains to see which have to be removed based on the update on extending chain
            if c1 == c2:
                continue

            c1_index, c2_index = self.find_mutual_block_index(c1, c2)  # find the fork point
            if c1_index == -1:
                continue

            if len(c2.miners) == 0:   # check to see if any miner is working on the c2 chain, if not remove it
                self.chains.remove(c2)
                del self.chains_headerblock[c2]
                for miner in c1.miners:
                    if isinstance(miner, UndercutMiner):
                        miner.infork = False
                        miner.first_block_fork = True
                continue

            if (c1.len() - c1_index) > self.conf_blocks:  # check to see if the extending chain has confirmed the fork point, if so remove all the fork chains
                for miner in c2.miners:
                    c1.miners.append(miner) # add the miner of c2 to c1
                    if isinstance(miner, UndercutMiner):
                        self.num_failed_forks += 1

                for miner in c1.miners: # reset the undercutting miners in the main chain
                    if isinstance(miner, UndercutMiner):
                        miner.infork = False
                        miner.first_block_fork = True
                c1.update_mining_powers()

                self.chains.remove(c2)
                del self.chains_headerblock[c2]


    def update_miners_chain(self, extending_chain):
        miner_change_chain = {}
        chain1 = extending_chain
        for chain2 in self.chains:
            if chain1 == chain2:
                continue

            c1_index, c2_index = self.find_mutual_block_index(chain1, chain2)
            if c1_index == -1:
                print("error in finding mutal block")
                exit()

            remaining_blocks_chain1 = self.conf_blocks - (chain1.len() - c1_index)
            remaining_blocks_chain2 = self.conf_blocks - (chain2.len() - c2_index)

            if remaining_blocks_chain1 >= self.conf_blocks - 1 or remaining_blocks_chain2 >= self.conf_blocks - 1:
                prob_chain1_wining = 0
            else:
                visible_blockrate_chain1 = chain1.get_visible_mining_power(c1_index) / self.avg_block_time_minutes
                visible_blockrate_chain2 = chain2.get_visible_mining_power(c2_index) / self.avg_block_time_minutes
                for miner in chain1.miners:
                    if isinstance(miner, UndercutMiner) and miner.infork:
                        visible_blockrate_chain1 = chain1.get_visible_mining_power(c1_index) / self.avg_block_time_minutes
                        visible_blockrate_chain2 = (1 - chain1.get_visible_mining_power(c1_index)) / self.avg_block_time_minutes
                for miner in chain2.miners:
                    if isinstance(miner, UndercutMiner) and miner.infork:
                        visible_blockrate_chain2 = chain2.get_visible_mining_power(c2_index) / self.avg_block_time_minutes
                        visible_blockrate_chain1 = (1 - chain2.get_visible_mining_power(c2_index)) / self.avg_block_time_minutes

                prob_chain1_wining = self.mnProduct(remaining_blocks_chain1, remaining_blocks_chain2,
                                                    visible_blockrate_chain1, visible_blockrate_chain2)
                try:
                    profit_blocks = min(chain1.len() - c1_index - 1, chain2.len() - c2_index - 1)
                    chain1_profits = chain1.get_profits(c1_index, c1_index + profit_blocks)
                    chain2_profits = chain2.get_profits(c2_index, c2_index + profit_blocks)
                    prob_chain1_wining *= (1 + (chain2_profits - chain1_profits) / max(chain1_profits, chain2_profits))
                except:
                    pass

            for miner in chain2.miners:
                if isinstance(miner, UndercutMiner) and miner.infork:
                    if remaining_blocks_chain2 - remaining_blocks_chain1 > self.undercutter_cutoff:
                        miner_change_chain[miner] = (chain2, chain1)
                        miner.infork = False
                        miner.first_block_fork = True
                        for miner in chain2.miners:
                            miner_change_chain[miner] = (chain2, chain1)
                        self.num_failed_forks += 1
                        self.num_failed_forks_cutoff += 1
                        continue
                    continue

                if isinstance(miner, HonestMiner):
                    if remaining_blocks_chain1 < remaining_blocks_chain2:  # chain1 is longer than chain2
                        miner_change_chain[miner] = (chain2, chain1)
                        continue
                    continue

                if random.random() < prob_chain1_wining:  # decide to change
                    miner_change_chain[miner] = (chain2, chain1)

        for miner, (fromchain, tochain) in miner_change_chain.items():
            tochain.miners.append(miner)
            fromchain.miners.remove(miner)
            fromchain.update_mining_powers()
            tochain.update_mining_powers()


    # ===========================================================================================================
    #   helper functions
    # ===========================================================================================================

    def find_mutual_block_index(self, chain1, chain2):
        i = -1
        j = -1
        while i >= -chain1.len() and j >= -chain2.len():
            try:
                if chain1.get_block(i) == chain2.get_block(j):
                    return i+chain1.len(),j+chain2.len()
                if chain1.get_block(i).height < chain2.get_block(j).height:
                    j -= 1
                elif chain1.get_block(i).height > chain2.get_block(j).height:
                    i -= 1
                else:
                    i -= 1
                    j -= 1
            except:
                print("error in finding the fork point")
                break
        print("Error in finding the mutual block, program exiting.")
        exit()



    def mnProduct(self, M=4, N=4, rate1=0, rate2=0):
        pr = integrate.dblquad(lambda x, y: math.pow(rate1, M + 1) * math.exp(-rate1 * x) * math.pow(x, M) *
                                            math.pow(rate2, N + 1) * math.exp(-rate2 * y) * math.pow(y, N) /
                                            math.factorial(M) / math.factorial(N), 0, np.inf, lambda x: 0, lambda x: x)
        return pr[0]


    def get_mempool_tx(self, prev_time, curr_time):
        new_tx = []
        for t in range(int(prev_time), int(curr_time)):
            try:
                for tx in all_txs[t]:
                    new_tx.append(Tx(timestamp=tx['timestamp'], size=tx['size'], fee=tx['fee'], serial=tx['serial']))
                    if tx['timestamp'] >= self.end_time:
                        self.experiment_end_flag = True
            except:
                pass
        return new_tx






######################################################################################################################



# Calculate the reward of the miners by going back through all the blocks in the last block of the main chain.
def get_miner_rewards(last_block):
    current_block = last_block  # working backwards
    undercutter_counter = 0
    block_counter = 0
    miner_reward = {}
    while current_block.parent_block != None:
        block_counter += 1
        m = current_block.miner
        if isinstance(m, UndercutMiner):
            undercutter_counter += 1
        reward = current_block.txfees
        try:
            miner_reward[m.name] += reward
        except:
            miner_reward[m.name] = reward

        current_block = current_block.parent_block

    return miner_reward, block_counter, undercutter_counter




########################################################################
# Setup for running the experiment for undercutter with 45% mining power
########################################################################
def setup_45(t,i,h,c):
    gc.collect()
    p=45
    print("45")
    print(t, i, h, c)
    time.sleep(random.random() * 10)

    # Create miners
    if h == 0:
        m0 = UndercutMiner(name=0, mining_power=0.45, mempool_threshold=t)
        miners = [m0]
        len_miners = len(miners)
        mining_powers = [.1, .1, .1, .1, .1, .05]
        for id, pow in enumerate(mining_powers):
            miners.append(NormalMiner(name=id + len_miners, mining_power=pow))


    elif h == 25:
        m0 = UndercutMiner(name=0, mining_power=0.45, mempool_threshold=t)
        m1 = HonestMiner(name=1, mining_power=0.25)
        miners = [m0, m1]
        len_miners = len(miners)
        mining_powers = [.1, .1, .1]
        for id, pow in enumerate(mining_powers):
            miners.append(NormalMiner(name=id + len_miners, mining_power=pow))


    elif h == 50:
        m0 = UndercutMiner(name=0, mining_power=0.45, mempool_threshold=t)
        m1 = HonestMiner(name=1, mining_power=0.1)
        m2 = HonestMiner(name=2, mining_power=0.1)
        m3 = HonestMiner(name=3, mining_power=0.1)
        m4 = HonestMiner(name=4, mining_power=0.1)
        m5 = HonestMiner(name=5, mining_power=0.1)
        miners = [m0, m1, m2, m3, m4, m5]
        len_miners = len(miners)
        mining_powers = [.05]
        for id, pow in enumerate(mining_powers):
            miners.append(NormalMiner(name=id + len_miners, mining_power=pow))

    # Create genesis block
    genesis_block = Block(None, [], begining_timestamp-1, None, 0, 0, 0)

    # Create simulation
    s = Simulation(genesis_block=genesis_block, miners=miners,
                   tx_file=tx_file_path, start_timestamp=begining_timestamp,
                   end_timestamp=ending_timestamp, t=t, i=i, cutoff=c)
    last_block, num_forks, num_failed_forks, num_failed_forks_cutoff, num_total_blocks = s.run()

    # Get results
    result, num_blocks, num_blocks_mined_undercutter = get_miner_rewards(last_block)
    f_out = open("results_final/cuttoff_{0}_undercutting_{1}percent_honest_{2}percent_leaveout_{3}_iteration_{4}".format(c,p,h,t,i), 'w')
    f_out.write(json.dumps(result) + "\n")
    f_out.write("num_forks: " + str(num_forks) + "\n")
    f_out.write("num_failed_forks: " + str(num_failed_forks) + "\n")
    f_out.write("num_cutoff_failed_forks: " + str(num_failed_forks_cutoff) + "\n")
    f_out.write("num_total_blocks: " + str(num_total_blocks) + "\n")
    f_out.write("num_blocks_mainchain: " + str(num_blocks) + "\n")
    f_out.write("num_blocks_mined_undercutter: " + str(num_blocks_mined_undercutter) + "\n")
    f_out.close()


########################################################################
# Setup for running the experiment for undercutter with 17% mining power
########################################################################
def setup_17(t,i,h,c):
    p = 17
    print(t, i, h, c)
    time.sleep(random.random() * 10)
    # Create miners
    if h == 0:
        m0 = UndercutMiner(name=0, mining_power=0.176, mempool_threshold=t)
        miners = [m0]
        len_miners = len(miners)
        mining_powers = [.153, .12, .118, .07, .067, .056, .056, .045, .043, .033, .027, .013, .01, .006, .006]
        for id, pow in enumerate(mining_powers):
            miners.append(NormalMiner(name=id + len_miners, mining_power=pow))


    elif h == 25:
        m0 = UndercutMiner(name=0, mining_power=0.176, mempool_threshold=t)
        m1 = HonestMiner(name=1, mining_power=.12)
        m2 = HonestMiner(name=2, mining_power=.118)
        m3 = HonestMiner(name=3, mining_power=.013) # sum= 0.251
        miners = [m0, m1, m2, m3]
        len_miners = len(miners)
        mining_powers = [.153, .07, .067, .056, .056, .045, .043, .033, .027, .01, .006, .006]
        for id, pow in enumerate(mining_powers):
            miners.append(NormalMiner(name=id + len_miners, mining_power=pow))


    elif h == 50:
        m0 = UndercutMiner(name=0, mining_power=0.172, mempool_threshold=t)
        m1 = HonestMiner(name=1, mining_power=.153)
        m2 = HonestMiner(name=2, mining_power=.12)
        m3 = HonestMiner(name=3, mining_power=.118)
        m4 = HonestMiner(name=4, mining_power=.067)
        m5 = HonestMiner(name=5, mining_power=.043)  # sum= 0.501
        miners = [m0, m1, m2, m3, m4, m5]
        len_miners = len(miners)
        mining_powers = [.07, .056, .056, .045, .033, .027, .013, .01, .006, .006]
        for id, pow in enumerate(mining_powers):
            miners.append(NormalMiner(name=id + len_miners, mining_power=pow))

    # Create genesis block
    genesis_block = Block(None, [], begining_timestamp-1, None, 0, 0, 0)

    # Create simulation
    s = Simulation(genesis_block=genesis_block, miners=miners,
                   tx_file=tx_file_path, start_timestamp=begining_timestamp,
                   end_timestamp=ending_timestamp, t=t, i=i, cutoff=c)
    last_block, num_forks, num_failed_forks, num_failed_forks_cutoff, num_total_blocks = s.run()

    # Get results
    result, num_blocks, num_blocks_mined_undercutter = get_miner_rewards(last_block)
    f_out = open("results_final/cuttoff_{0}_undercutting_{1}percent_honest_{2}percent_leaveout_{3}_iteration_{4}".format(c,p,h,t,i), 'w')
    f_out.write(json.dumps(result) + "\n")
    f_out.write("num_forks: " + str(num_forks) + "\n")
    f_out.write("num_failed_forks: " + str(num_failed_forks) + "\n")
    f_out.write("num_cutoff_failed_forks: " + str(num_failed_forks_cutoff) + "\n")
    f_out.write("num_total_blocks: " + str(num_total_blocks) + "\n")
    f_out.write("num_blocks_mainchain: " + str(num_blocks) + "\n")
    f_out.write("num_blocks_mined_undercutter: " + str(num_blocks_mined_undercutter) + "\n")
    f_out.close()




#################################################################################################
#                           MAIN
#################################################################################################
if __name__== "__main__":

    load_txs(tx_file_path)
    undercutting_thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    iterations = range(1,2,1)
    honest_percentage = [25, 50]
    cutoff = [1, 10]


    paramlist_list = list(itertools.product(undercutting_thresholds, iterations, honest_percentage, cutoff))
    print(paramlist_list)
    print(len(paramlist_list))

    p = Pool(10)
    p.starmap(setup_45, paramlist_list)
    p.close()
    p.join()
    print("done")


































































#
# def setup_0honest_40(t,i):
#
#     # # Create miners
#     m0 = UndercutMiner(name=0, mining_power=0.4, mempool_threshold=t)
#     miners = [m0]
#
#     len_miners = len(miners)
#     mining_powers = [.1,.1,.1,.1,.1,.1]
#     for id, pow in enumerate(mining_powers):
#         miners.append(NormalMiner(name=id + len_miners , mining_power=pow))
#
#     # Create genesis block
#     genesis_block = Block(None, [], 1577835687, None, 0, 0, 0)
#     #genesis_block = Block(None, [], 0, None, 0, 0, 0)
#
#     # Create simulation
#     #s = Simulation(genesis_block=genesis_block, miners=miners, tx_file='data/txs', start_timestamp=0, end_timestamp=100, t=t, i=i)
#     s = Simulation(genesis_block=genesis_block, miners=miners,
#                    tx_file='/4tbssd/undercutting_project/summary_txs_2020_Feb', start_timestamp=1577835688,
#                    end_timestamp=1583019688, t=t, i=i)  # real_end_timestamp = 1581292777
#
#     last_block, num_forks = s.run()
#     result = get_miner_rewards(last_block)
#
#     f_out = open("results/undercutting_1miner_40percent_honest_0miner_0percent_leaveout_{0}_iteration_{1}".format(t,i), 'w')
#     f_out.write(json.dumps(result)+"\n")
#     f_out.write("num_forks:" + str(num_forks) + "\n")
#     f_out.flush()
#     f_out.close()
#
#
#
# def setup_10honest_40(t,i):
#
#     # # Create miners
#     m0 = UndercutMiner(name=0, mining_power=0.4, mempool_threshold=t)
#     m1 = HonestMiner(name=1, mining_power=0.1)
#     miners = [m0,m1]
#
#     len_miners = len(miners)
#     mining_powers = [.1,.1,.1,.1,.1]
#     for id, pow in enumerate(mining_powers):
#         miners.append(NormalMiner(name=id + len_miners , mining_power=pow))
#
#     # Create genesis block
#     genesis_block = Block(None, [], 1577835687, None, 0, 0, 0)
#     #genesis_block = Block(None, [], 0, None, 0, 0, 0)
#
#     # Create simulation
#     #s = Simulation(genesis_block=genesis_block, miners=miners, tx_file='data/txs', start_timestamp=0, end_timestamp=100, t=t, i=i)
#     s = Simulation(genesis_block=genesis_block, miners=miners,
#                    tx_file='/4tbssd/undercutting_project/summary_txs_2020_Feb', start_timestamp=1577835688,
#                    end_timestamp=1583019688, t=t, i=i)  # real_end_timestamp = 1581292777
#
#     last_block, num_forks = s.run()
#     result = get_miner_rewards(last_block)
#
#     f_out = open("results/undercutting_1miner_40percent_honest_1miner_10percent_leaveout_{0}_iteration_{1}".format(t,i), 'w')
#     f_out.write(json.dumps(result)+"\n")
#     f_out.write("num_forks:" + str(num_forks) + "\n")
#     f_out.flush()
#     f_out.close()
#
#
#
# def setup_30honest_40(t,i):
#
#     # # Create miners
#     m0 = UndercutMiner(name=0, mining_power=0.4, mempool_threshold=t)
#     m1 = HonestMiner(name=1, mining_power=0.1)
#     m2 = HonestMiner(name=2, mining_power=0.1)
#     m3 = HonestMiner(name=3, mining_power=0.1)
#     miners = [m0,m1,m2,m3]
#
#     len_miners = len(miners)
#     mining_powers = [.1,.1,.1]
#     for id, pow in enumerate(mining_powers):
#         miners.append(NormalMiner(name=id + len_miners , mining_power=pow))
#
#     # Create genesis block
#     genesis_block = Block(None, [], 1577835687, None, 0, 0, 0)
#     #genesis_block = Block(None, [], 0, None, 0, 0, 0)
#
#     # Create simulation
#     #s = Simulation(genesis_block=genesis_block, miners=miners, tx_file='data/txs', start_timestamp=0, end_timestamp=100, t=t, i=i)
#     s = Simulation(genesis_block=genesis_block, miners=miners,
#                    tx_file='/4tbssd/undercutting_project/summary_txs_2020_Feb', start_timestamp=1577835688,
#                    end_timestamp=1583019688, t=t, i=i)  # real_end_timestamp = 1581292777
#
#     last_block, num_forks = s.run()
#     result = get_miner_rewards(last_block)
#
#     f_out = open("results/undercutting_1miner_40percent_honest_3miner_30percent_leaveout_{0}_iteration_{1}".format(t,i), 'w')
#     f_out.write(json.dumps(result)+"\n")
#     f_out.write("num_forks:" + str(num_forks) + "\n")
#     f_out.flush()
#     f_out.close()
#
#
#
# def setup_50honest_40(t,i):
#
#     # # Create miners
#     m0 = UndercutMiner(name=0, mining_power=0.4, mempool_threshold=t)
#     m1 = HonestMiner(name=1, mining_power=0.1)
#     m2 = HonestMiner(name=2, mining_power=0.1)
#     m3 = HonestMiner(name=3, mining_power=0.1)
#     m4 = HonestMiner(name=4, mining_power=0.1)
#     m5 = HonestMiner(name=5, mining_power=0.1)
#     miners = [m0,m1,m2,m3,m4,m5]
#
#     len_miners = len(miners)
#     mining_powers = [.1]
#     for id, pow in enumerate(mining_powers):
#         miners.append(NormalMiner(name=id + len_miners , mining_power=pow))
#
#     # Create genesis block
#     genesis_block = Block(None, [], 1577835687, None, 0, 0, 0)
#     #genesis_block = Block(None, [], 0, None, 0, 0, 0)
#
#     # Create simulation
#     #s = Simulation(genesis_block=genesis_block, miners=miners, tx_file='data/txs', start_timestamp=0, end_timestamp=100, t=t, i=i)
#     s = Simulation(genesis_block=genesis_block, miners=miners,
#                    tx_file='/4tbssd/undercutting_project/summary_txs_2020_Feb', start_timestamp=1577835688,
#                    end_timestamp=1583019688, t=t, i=i)  # real_end_timestamp = 1581292777
#
#     last_block, num_forks = s.run()
#     result = get_miner_rewards(last_block)
#
#     f_out = open("results/undercutting_1miner_40percent_honest_5miner_50percent_leaveout_{0}_iteration_{1}".format(t,i), 'w')
#     f_out.write(json.dumps(result)+"\n")
#     f_out.write("num_forks:" + str(num_forks) + "\n")
#     f_out.flush()
#     f_out.close()
#
#
# ##########################################################
# ##########################################################
# ##########################################################
# def setup_0honest_45(t,i):
#
#     # # Create miners
#     m0 = UndercutMiner(name=0, mining_power=0.45, mempool_threshold=t)
#     miners = [m0]
#
#     len_miners = len(miners)
#     mining_powers = [.1,.1,.1,.1,.1,.05]
#     for id, pow in enumerate(mining_powers):
#         miners.append(NormalMiner(name=id + len_miners , mining_power=pow))
#
#     # Create genesis block
#     genesis_block = Block(None, [], 1577835687, None, 0, 0, 0)
#     #genesis_block = Block(None, [], 0, None, 0, 0, 0)
#
#     # Create simulation
#     #s = Simulation(genesis_block=genesis_block, miners=miners, tx_file='data/txs', start_timestamp=0, end_timestamp=100, t=t, i=i)
#     s = Simulation(genesis_block=genesis_block, miners=miners,
#                    tx_file='/4tbssd/undercutting_project/summary_txs_2020_Feb', start_timestamp=1577835688,
#                    end_timestamp=1582019688, t=t, i=i)  # real_end_timestamp = 1581292777
#
#     last_block, num_forks = s.run()
#     result = get_miner_rewards(last_block)
#
#     f_out = open("results2/stubborn_undercutting_1miner_45percent_honest_0miner_0percent_leaveout_{0}_iteration_{1}".format(t,i), 'w')
#     f_out.write(json.dumps(result)+"\n")
#     f_out.write("num_forks:" + str(num_forks) + "\n")
#     f_out.flush()
#     f_out.close()
#
#
#
# def setup_10honest_45(t,i):
#     print("10")
#     time.sleep(2)
#     # # Create miners
#     m0 = UndercutMiner(name=0, mining_power=0.45, mempool_threshold=t)
#     m1 = HonestMiner(name=1, mining_power=0.1)
#     miners = [m0,m1]
#
#     len_miners = len(miners)
#     mining_powers = [.1,.1,.1,.1,.05]
#     for id, pow in enumerate(mining_powers):
#         miners.append(NormalMiner(name=id + len_miners , mining_power=pow))
#
#     # Create genesis block
#     genesis_block = Block(None, [], 1577835687, None, 0, 0, 0)
#     #genesis_block = Block(None, [], 0, None, 0, 0, 0)
#
#     # Create simulation
#     #s = Simulation(genesis_block=genesis_block, miners=miners, tx_file='data/txs', start_timestamp=0, end_timestamp=100, t=t, i=i)
#     s = Simulation(genesis_block=genesis_block, miners=miners,
#                    tx_file='/4tbssd/undercutting_project/summary_txs_2020_Feb', start_timestamp=1577835688,
#                    end_timestamp=1582019688, t=t, i=i)  # real_end_timestamp = 1581292777
#
#     last_block, num_forks = s.run()
#     result = get_miner_rewards(last_block)
#
#     f_out = open("results2/stubborn_undercutting_1miner_45percent_honest_1miner_10percent_leaveout_{0}_iteration_{1}".format(t,i), 'w')
#     f_out.write(json.dumps(result)+"\n")
#     f_out.write("num_forks:" + str(num_forks) + "\n")
#     f_out.flush()
#     f_out.close()
#
#
#
# def setup_30honest_45(t,i):
#
#     # # Create miners
#     m0 = UndercutMiner(name=0, mining_power=0.45, mempool_threshold=t)
#     m1 = HonestMiner(name=1, mining_power=0.1)
#     m2 = HonestMiner(name=2, mining_power=0.1)
#     m3 = HonestMiner(name=3, mining_power=0.1)
#     miners = [m0,m1,m2,m3]
#
#     len_miners = len(miners)
#     mining_powers = [.1,.1,.05]
#     for id, pow in enumerate(mining_powers):
#         miners.append(NormalMiner(name=id + len_miners , mining_power=pow))
#
#     # Create genesis block
#     genesis_block = Block(None, [], 1577835687, None, 0, 0, 0)
#     #genesis_block = Block(None, [], 0, None, 0, 0, 0)
#
#     # Create simulation
#     #s = Simulation(genesis_block=genesis_block, miners=miners, tx_file='data/txs', start_timestamp=0, end_timestamp=100, t=t, i=i)
#     s = Simulation(genesis_block=genesis_block, miners=miners,
#                    tx_file='/4tbssd/undercutting_project/summary_txs_2020_Feb', start_timestamp=1577835688,
#                    end_timestamp=1582019688, t=t, i=i)  # real_end_timestamp = 1581292777
#
#     last_block, num_forks = s.run()
#     result = get_miner_rewards(last_block)
#
#     f_out = open("results2/stubborn_undercutting_1miner_45percent_honest_3miner_30percent_leaveout_{0}_iteration_{1}".format(t,i), 'w')
#     f_out.write(json.dumps(result)+"\n")
#     f_out.write("num_forks:" + str(num_forks) + "\n")
#     f_out.flush()
#     f_out.close()
#
#
#
# def setup_50honest_45(t,i):
#
#     # # Create miners
#     m0 = UndercutMiner(name=0, mining_power=0.45, mempool_threshold=t)
#     m1 = HonestMiner(name=1, mining_power=0.1)
#     m2 = HonestMiner(name=2, mining_power=0.1)
#     m3 = HonestMiner(name=3, mining_power=0.1)
#     m4 = HonestMiner(name=4, mining_power=0.1)
#     m5 = HonestMiner(name=5, mining_power=0.1)
#     miners = [m0,m1,m2,m3,m4,m5]
#
#     len_miners = len(miners)
#     mining_powers = [.05]
#     for id, pow in enumerate(mining_powers):
#         miners.append(NormalMiner(name=id + len_miners , mining_power=pow))
#
#     # Create genesis block
#     genesis_block = Block(None, [], 1577835687, None, 0, 0, 0)
#     #genesis_block = Block(None, [], 0, None, 0, 0, 0)
#
#     # Create simulation
#     #s = Simulation(genesis_block=genesis_block, miners=miners, tx_file='data/txs', start_timestamp=0, end_timestamp=100, t=t, i=i)
#     s = Simulation(genesis_block=genesis_block, miners=miners,
#                    tx_file='/4tbssd/undercutting_project/summary_txs_2020_Feb', start_timestamp=1577835688,
#                    end_timestamp=1582019688, t=t, i=i)  # real_end_timestamp = 1581292777
#
#     last_block, num_forks = s.run()
#     result = get_miner_rewards(last_block)
#
#     f_out = open("results2/stubborn_undercutting_1miner_45percent_honest_5miner_50percent_leaveout_{0}_iteration_{1}".format(t,i), 'w')
#     f_out.write(json.dumps(result)+"\n")
#     f_out.write("num_forks:" + str(num_forks) + "\n")
#     f_out.flush()
#     f_out.close()
#
# ##########################################################
# ##########################################################
# ##########################################################
#
# def setup_0honest_trueminingpower(t,i):
#     # # Create miners
#     m0 = UndercutMiner(name=0, mining_power=0.172, mempool_threshold=t)
#     miners = [m0]
#
#     len_miners = len(miners)
#     mining_powers = [.156, .116, .116, .077, .075, .058, .044, .042, .037, .035, .026, .018, .012, .007, .005, .004]
#     for id, pow in enumerate(mining_powers):
#         miners.append(NormalMiner(name=id + len_miners , mining_power=pow))
#
#     # Create genesis block
#     genesis_block = Block(None, [], 1577835687, None, 0, 0, 0)
#     #genesis_block = Block(None, [], 0, None, 0, 0, 0)
#
#     # Create simulation
#     #s = Simulation(genesis_block=genesis_block, miners=miners, tx_file='data/txs', start_timestamp=0, end_timestamp=100, t=t, i=i)
#     s = Simulation(genesis_block=genesis_block, miners=miners,
#                    tx_file='/4tbssd/undercutting_project/summary_txs_2020_Feb', start_timestamp=1577835688,
#                    end_timestamp=1582019688, t=t, i=i)  # real_end_timestamp = 1581292777
#
#     last_block, num_forks = s.run()
#     result = get_miner_rewards(last_block)
#
#     f_out = open("results2/stubborn_undercutting_1miner_17percent_honest_0miner_0percent_leaveout_{0}_iteration_{1}".format(t,i), 'w')
#     f_out.write(json.dumps(result)+"\n")
#     f_out.write("num_forks:" + str(num_forks) + "\n")
#     f_out.flush()
#     f_out.close()
#
#
#
# def setup_10honest_trueminingpower(t,i):
#     # # Create miners
#     m0 = UndercutMiner(name=0, mining_power=0.172, mempool_threshold=t)
#     m1 = HonestMiner(name = 1, mining_power=.075)
#     m2 = HonestMiner(name= 2, mining_power= 0.026) #sum= 0.101
#     miners = [m0, m1, m2]
#
#     len_miners = len(miners)
#     mining_powers = [.156, .116, .116, .077, .058, .044, .042, .037, .035, .018, .012, .007, .005, .004]
#     for id, pow in enumerate(mining_powers):
#         miners.append(NormalMiner(name=id + len_miners , mining_power=pow))
#
#     # Create genesis block
#     genesis_block = Block(None, [], 1577835687, None, 0, 0, 0)
#     #genesis_block = Block(None, [], 0, None, 0, 0, 0)
#
#     # Create simulation
#     #s = Simulation(genesis_block=genesis_block, miners=miners, tx_file='data/txs', start_timestamp=0, end_timestamp=100, t=t, i=i)
#     s = Simulation(genesis_block=genesis_block, miners=miners,
#                    tx_file='/4tbssd/undercutting_project/summary_txs_2020_Feb', start_timestamp=1577835688,
#                    end_timestamp=1582019688, t=t, i=i)  # real_end_timestamp = 1581292777
#
#     last_block, num_forks = s.run()
#     result = get_miner_rewards(last_block)
#
#     f_out = open("results2/stubborn_undercutting_1miner_17percent_honest_2miner_10percent_leaveout_{0}_iteration_{1}".format(t,i), 'w')
#     f_out.write(json.dumps(result)+"\n")
#     f_out.write("num_forks:" + str(num_forks) + "\n")
#     f_out.flush()
#     f_out.close()
#
#
# def setup_30honest_trueminingpower(t,i):
#     # # Create miners
#     m0 = UndercutMiner(name=0, mining_power=0.172, mempool_threshold=t)
#     m1 = HonestMiner(name= 1, mining_power= .075)
#     m2 = HonestMiner(name= 2, mining_power= .026)
#     m3 = HonestMiner(name= 3, mining_power= .156)
#     m4 = HonestMiner(name= 4, mining_power= .042) #sum= 0.299
#     miners = [m0, m1, m2, m3, m4]
#
#     len_miners = len(miners)
#     mining_powers = [.116, .116, .077, .058, .044, .037, .035, .018, .012, .007, .005, .004]
#     for id, pow in enumerate(mining_powers):
#         miners.append(NormalMiner(name=id + len_miners , mining_power=pow))
#
#     # Create genesis block
#     genesis_block = Block(None, [], 1577835687, None, 0, 0, 0)
#     #genesis_block = Block(None, [], 0, None, 0, 0, 0)
#
#     # Create simulation
#     #s = Simulation(genesis_block=genesis_block, miners=miners, tx_file='data/txs', start_timestamp=0, end_timestamp=100, t=t, i=i)
#     s = Simulation(genesis_block=genesis_block, miners=miners,
#                    tx_file='/4tbssd/undercutting_project/summary_txs_2020_Feb', start_timestamp=1577835688,
#                    end_timestamp=1582019688, t=t, i=i)  # real_end_timestamp = 1581292777
#
#     last_block, num_forks = s.run()
#     result = get_miner_rewards(last_block)
#
#     f_out = open("results2/stubborn_undercutting_1miner_17percent_honest_4miner_30percent_leaveout_{0}_iteration_{1}".format(t,i), 'w')
#     f_out.write(json.dumps(result)+"\n")
#     f_out.write("num_forks:" + str(num_forks) + "\n")
#     f_out.flush()
#     f_out.close()
#
#
#
# def setup_50honest_trueminingpower(t,i):
#     # # Create miners
#     m0 = UndercutMiner(name=0, mining_power=0.172, mempool_threshold=t)
#     m1 = HonestMiner(name= 1, mining_power= .075)
#     m2 = HonestMiner(name= 2, mining_power= .026)
#     m3 = HonestMiner(name= 3, mining_power= .156)
#     m4 = HonestMiner(name= 4, mining_power= .042)
#     m5 = HonestMiner(name= 5, mining_power= .116)
#     m6 = HonestMiner(name= 6, mining_power= .077)
#     m7 = HonestMiner(name= 7, mining_power= .007) #sum= 0.499
#     miners = [m0, m1, m2, m3, m4, m5, m6, m7]
#
#     len_miners = len(miners)
#     mining_powers = [.116, .058, .044, .037, .035, .018, .012, .005, .004]
#     for id, pow in enumerate(mining_powers):
#         miners.append(NormalMiner(name=id + len_miners , mining_power=pow))
#
#     # Create genesis block
#     genesis_block = Block(None, [], 1577835687, None, 0, 0, 0)
#     #genesis_block = Block(None, [], 0, None, 0, 0, 0)
#
#     # Create simulation
#     #s = Simulation(genesis_block=genesis_block, miners=miners, tx_file='data/txs', start_timestamp=0, end_timestamp=100, t=t, i=i)
#     s = Simulation(genesis_block=genesis_block, miners=miners,
#                    tx_file='/4tbssd/undercutting_project/summary_txs_2020_Feb', start_timestamp=1577835688,
#                    end_timestamp=1582019688, t=t, i=i)  # real_end_timestamp = 1581292777
#
#     last_block, num_forks = s.run()
#     result = get_miner_rewards(last_block)
#
#     f_out = open("results2/stubborn_undercutting_1miner_17percent_honest_7miner_50percent_leaveout_{0}_iteration_{1}".format(t,i), 'w')
#     f_out.write(json.dumps(result)+"\n")
#     f_out.write("num_forks:" + str(num_forks) + "\n")
#     f_out.flush()
#     f_out.close()
#
# ##########################################################
# ##########################################################
# ##########################################################
#
# def setup_0honest_30(t,i):
#
#     # # Create miners
#     m0 = UndercutMiner(name=0, mining_power=0.3, mempool_threshold=t)
#     miners = [m0]
#
#     len_miners = len(miners)
#     mining_powers = [.1,.1,.1,.1,.1,.1,.1]
#     for id, pow in enumerate(mining_powers):
#         miners.append(NormalMiner(name=id + len_miners , mining_power=pow))
#
#     # Create genesis block
#     genesis_block = Block(None, [], 1577835687, None, 0, 0, 0)
#     #genesis_block = Block(None, [], 0, None, 0, 0, 0)
#
#     # Create simulation
#     #s = Simulation(genesis_block=genesis_block, miners=miners, tx_file='data/txs', start_timestamp=0, end_timestamp=100, t=t, i=i)
#     s = Simulation(genesis_block=genesis_block, miners=miners,
#                    tx_file='/4tbssd/undercutting_project/summary_txs_2020_Feb', start_timestamp=1577835688,
#                    end_timestamp=1583019688, t=t, i=i)  # real_end_timestamp = 1581292777
#
#     last_block, num_forks = s.run()
#     result = get_miner_rewards(last_block)
#
#     f_out = open("results/undercutting_1miner_30percent_honest_0miner_0percent_leaveout_{0}_iteration_{1}".format(t,i), 'w')
#     f_out.write(json.dumps(result)+"\n")
#     f_out.write("num_forks:" + str(num_forks) + "\n")
#     f_out.flush()
#     f_out.close()
#
#
#
# def setup_10honest_30(t,i):
#
#     # # Create miners
#     m0 = UndercutMiner(name=0, mining_power=0.3, mempool_threshold=t)
#     m1 = HonestMiner(name=1, mining_power=0.1)
#     miners = [m0,m1]
#
#     len_miners = len(miners)
#     mining_powers = [.1,.1,.1,.1,.1,.1]
#     for id, pow in enumerate(mining_powers):
#         miners.append(NormalMiner(name=id + len_miners , mining_power=pow))
#
#     # Create genesis block
#     genesis_block = Block(None, [], 1577835687, None, 0, 0, 0)
#     #genesis_block = Block(None, [], 0, None, 0, 0, 0)
#
#     # Create simulation
#     #s = Simulation(genesis_block=genesis_block, miners=miners, tx_file='data/txs', start_timestamp=0, end_timestamp=100, t=t, i=i)
#     s = Simulation(genesis_block=genesis_block, miners=miners,
#                    tx_file='/4tbssd/undercutting_project/summary_txs_2020_Feb', start_timestamp=1577835688,
#                    end_timestamp=1583019688, t=t, i=i)  # real_end_timestamp = 1581292777
#
#     last_block, num_forks = s.run()
#     result = get_miner_rewards(last_block)
#
#     f_out = open("results/undercutting_1miner_30percent_honest_1miner_10percent_leaveout_{0}_iteration_{1}".format(t,i), 'w')
#     f_out.write(json.dumps(result)+"\n")
#     f_out.write("num_forks:" + str(num_forks) + "\n")
#     f_out.flush()
#     f_out.close()
#
#
#
# def setup_30honest_30(t,i):
#
#     # # Create miners
#     m0 = UndercutMiner(name=0, mining_power=0.3, mempool_threshold=t)
#     m1 = HonestMiner(name=1, mining_power=0.1)
#     m2 = HonestMiner(name=2, mining_power=0.1)
#     m3 = HonestMiner(name=3, mining_power=0.1)
#     miners = [m0,m1,m2,m3]
#
#     len_miners = len(miners)
#     mining_powers = [.1,.1,.1,.1]
#     for id, pow in enumerate(mining_powers):
#         miners.append(NormalMiner(name=id + len_miners , mining_power=pow))
#
#     # Create genesis block
#     genesis_block = Block(None, [], 1577835687, None, 0, 0, 0)
#     #genesis_block = Block(None, [], 0, None, 0, 0, 0)
#
#     # Create simulation
#     #s = Simulation(genesis_block=genesis_block, miners=miners, tx_file='data/txs', start_timestamp=0, end_timestamp=100, t=t, i=i)
#     s = Simulation(genesis_block=genesis_block, miners=miners,
#                    tx_file='/4tbssd/undercutting_project/summary_txs_2020_Feb', start_timestamp=1577835688,
#                    end_timestamp=1583019688, t=t, i=i)  # real_end_timestamp = 1581292777
#
#     last_block, num_forks = s.run()
#     result = get_miner_rewards(last_block)
#
#     f_out = open("results/undercutting_1miner_30percent_honest_3miner_30percent_leaveout_{0}_iteration_{1}".format(t,i), 'w')
#     f_out.write(json.dumps(result)+"\n")
#     f_out.write("num_forks:" + str(num_forks) + "\n")
#     f_out.flush()
#     f_out.close()
#
#
#
# def setup_50honest_30(t,i):
#
#     # # Create miners
#     m0 = UndercutMiner(name=0, mining_power=0.3, mempool_threshold=t)
#     m1 = HonestMiner(name=1, mining_power=0.1)
#     m2 = HonestMiner(name=2, mining_power=0.1)
#     m3 = HonestMiner(name=3, mining_power=0.1)
#     m4 = HonestMiner(name=4, mining_power=0.1)
#     m5 = HonestMiner(name=5, mining_power=0.1)
#     miners = [m0,m1,m2,m3,m4,m5]
#
#     len_miners = len(miners)
#     mining_powers = [.1,.1]
#     for id, pow in enumerate(mining_powers):
#         miners.append(NormalMiner(name=id + len_miners , mining_power=pow))
#
#     # Create genesis block
#     genesis_block = Block(None, [], 1577835687, None, 0, 0, 0)
#     #genesis_block = Block(None, [], 0, None, 0, 0, 0)
#
#     # Create simulation
#     #s = Simulation(genesis_block=genesis_block, miners=miners, tx_file='data/txs', start_timestamp=0, end_timestamp=100, t=t, i=i)
#     s = Simulation(genesis_block=genesis_block, miners=miners,
#                    tx_file='/4tbssd/undercutting_project/summary_txs_2020_Feb', start_timestamp=1577835688,
#                    end_timestamp=1583019688, t=t, i=i)  # real_end_timestamp = 1581292777
#
#     last_block, num_forks = s.run()
#     result = get_miner_rewards(last_block)
#
#     f_out = open("results/undercutting_1miner_30percent_honest_5miner_50percent_leaveout_{0}_iteration_{1}".format(t,i), 'w')
#     f_out.write(json.dumps(result)+"\n")
#     f_out.write("num_forks:" + str(num_forks) + "\n")
#     f_out.flush()
#     f_out.close()
#
#
# ##########################################################
# ##########################################################
# ##########################################################
#
# def setup_0honest_trueminingpower_nonstubborn(t,i):
#     print("0honest_trueminingpower_nonstubborn")
#     time.sleep(2)
#     # # Create miners
#     m0 = UndercutMiner(name=0, mining_power=0.172, mempool_threshold=t)
#     miners = [m0]
#
#     len_miners = len(miners)
#     mining_powers = [.156, .116, .116, .077, .075, .058, .044, .042, .037, .035, .026, .018, .012, .007, .005, .004]
#     for id, pow in enumerate(mining_powers):
#         miners.append(NormalMiner(name=id + len_miners , mining_power=pow))
#
#     # Create genesis block
#     genesis_block = Block(None, [], 1577835687, None, 0, 0, 0)
#     #genesis_block = Block(None, [], 0, None, 0, 0, 0)
#
#     # Create simulation
#     #s = Simulation(genesis_block=genesis_block, miners=miners, tx_file='data/txs', start_timestamp=0, end_timestamp=100, t=t, i=i)
#     s = Simulation(genesis_block=genesis_block, miners=miners,
#                    tx_file='/4tbssd/undercutting_project/summary_txs_2020_Feb', start_timestamp=1577835688,
#                    end_timestamp=1582019688, t=t, i=i)  # real_end_timestamp = 1581292777
#
#     last_block, num_forks = s.run()
#     result = get_miner_rewards(last_block)
#
#     f_out = open("results2/nonstubborn3_undercutting_1miner_17percent_honest_0miner_0percent_leaveout_{0}_iteration_{1}".format(t,i), 'w')
#     f_out.write(json.dumps(result)+"\n")
#     f_out.write("num_forks:" + str(num_forks) + "\n")
#     f_out.flush()
#     f_out.close()
#
#
#
# def setup_10honest_trueminingpower_nonstubborn(t,i):
#     print("10honest_trueminingpower_nonstubborn")
#     time.sleep(2)
#     # # Create miners
#     m0 = UndercutMiner(name=0, mining_power=0.172, mempool_threshold=t)
#     m1 = HonestMiner(name = 1, mining_power=.075)
#     m2 = HonestMiner(name= 2, mining_power= 0.026) #sum= 0.101
#     miners = [m0, m1, m2]
#
#     len_miners = len(miners)
#     mining_powers = [.156, .116, .116, .077, .058, .044, .042, .037, .035, .018, .012, .007, .005, .004]
#     for id, pow in enumerate(mining_powers):
#         miners.append(NormalMiner(name=id + len_miners , mining_power=pow))
#
#     # Create genesis block
#     genesis_block = Block(None, [], 1577835687, None, 0, 0, 0)
#     #genesis_block = Block(None, [], 0, None, 0, 0, 0)
#
#     # Create simulation
#     #s = Simulation(genesis_block=genesis_block, miners=miners, tx_file='data/txs', start_timestamp=0, end_timestamp=100, t=t, i=i)
#     s = Simulation(genesis_block=genesis_block, miners=miners,
#                    tx_file='/4tbssd/undercutting_project/summary_txs_2020_Feb', start_timestamp=1577835688,
#                    end_timestamp=1582019688, t=t, i=i)  # real_end_timestamp = 1581292777
#
#     last_block, num_forks = s.run()
#     result = get_miner_rewards(last_block)
#
#     f_out = open("results2/nonstubborn3_undercutting_1miner_17percent_honest_2miner_10percent_leaveout_{0}_iteration_{1}".format(t,i), 'w')
#     f_out.write(json.dumps(result)+"\n")
#     f_out.write("num_forks:" + str(num_forks) + "\n")
#     f_out.flush()
#     f_out.close()
#
#
# def setup_30honest_trueminingpower_nonstubborn(t,i):
#     print("30honest_trueminingpower_nonstubborn")
#     time.sleep(2)
#     # # Create miners
#     m0 = UndercutMiner(name=0, mining_power=0.172, mempool_threshold=t)
#     m1 = HonestMiner(name= 1, mining_power= .075)
#     m2 = HonestMiner(name= 2, mining_power= .026)
#     m3 = HonestMiner(name= 3, mining_power= .156)
#     m4 = HonestMiner(name= 4, mining_power= .042) #sum= 0.299
#     miners = [m0, m1, m2, m3, m4]
#
#     len_miners = len(miners)
#     mining_powers = [.116, .116, .077, .058, .044, .037, .035, .018, .012, .007, .005, .004]
#     for id, pow in enumerate(mining_powers):
#         miners.append(NormalMiner(name=id + len_miners , mining_power=pow))
#
#     # Create genesis block
#     genesis_block = Block(None, [], 1577835687, None, 0, 0, 0)
#     #genesis_block = Block(None, [], 0, None, 0, 0, 0)
#
#     # Create simulation
#     #s = Simulation(genesis_block=genesis_block, miners=miners, tx_file='data/txs', start_timestamp=0, end_timestamp=100, t=t, i=i)
#     s = Simulation(genesis_block=genesis_block, miners=miners,
#                    tx_file='/4tbssd/undercutting_project/summary_txs_2020_Feb', start_timestamp=1577835688,
#                    end_timestamp=1582019688, t=t, i=i)  # real_end_timestamp = 1581292777
#
#     last_block, num_forks = s.run()
#     result = get_miner_rewards(last_block)
#
#     f_out = open("results2/nonstubborn3_undercutting_1miner_17percent_honest_4miner_30percent_leaveout_{0}_iteration_{1}".format(t,i), 'w')
#     f_out.write(json.dumps(result)+"\n")
#     f_out.write("num_forks:" + str(num_forks) + "\n")
#     f_out.flush()
#     f_out.close()
#
#
#
# def setup_50honest_trueminingpower_nonstubborn(t,i):
#     print("50honest_trueminingpower_nonstubborn")
#     time.sleep(2)
#     # # Create miners
#     m0 = UndercutMiner(name=0, mining_power=0.172, mempool_threshold=t)
#     m1 = HonestMiner(name= 1, mining_power= .075)
#     m2 = HonestMiner(name= 2, mining_power= .026)
#     m3 = HonestMiner(name= 3, mining_power= .156)
#     m4 = HonestMiner(name= 4, mining_power= .042)
#     m5 = HonestMiner(name= 5, mining_power= .116)
#     m6 = HonestMiner(name= 6, mining_power= .077)
#     m7 = HonestMiner(name= 7, mining_power= .007) #sum= 0.499
#     miners = [m0, m1, m2, m3, m4, m5, m6, m7]
#
#     len_miners = len(miners)
#     mining_powers = [.116, .058, .044, .037, .035, .018, .012, .005, .004]
#     for id, pow in enumerate(mining_powers):
#         miners.append(NormalMiner(name=id + len_miners , mining_power=pow))
#
#     # Create genesis block
#     genesis_block = Block(None, [], 1577835687, None, 0, 0, 0)
#     #genesis_block = Block(None, [], 0, None, 0, 0, 0)
#
#     # Create simulation
#     #s = Simulation(genesis_block=genesis_block, miners=miners, tx_file='data/txs', start_timestamp=0, end_timestamp=100, t=t, i=i)
#     s = Simulation(genesis_block=genesis_block, miners=miners,
#                    tx_file='/4tbssd/undercutting_project/summary_txs_2020_Feb', start_timestamp=1577835688,
#                    end_timestamp=1582019688, t=t, i=i)  # real_end_timestamp = 1581292777
#
#     last_block, num_forks = s.run()
#     result = get_miner_rewards(last_block)
#
#     f_out = open("results2/nonstubborn3_undercutting_1miner_17percent_honest_7miner_50percent_leaveout_{0}_iteration_{1}".format(t,i), 'w')
#     f_out.write(json.dumps(result)+"\n")
#     f_out.write("num_forks:" + str(num_forks) + "\n")
#     f_out.flush()
#     f_out.close()
#
#
#
# ##########################################################
# ##########################################################
# ##########################################################
#
# def setup_0honest_40_nonstubborn(t,i):
#
#     # # Create miners
#     m0 = UndercutMiner(name=0, mining_power=0.4, mempool_threshold=t)
#     miners = [m0]
#
#     len_miners = len(miners)
#     mining_powers = [.1,.1,.1,.1,.1,.1]
#     for id, pow in enumerate(mining_powers):
#         miners.append(NormalMiner(name=id + len_miners , mining_power=pow))
#
#     # Create genesis block
#     genesis_block = Block(None, [], 1577835687, None, 0, 0, 0)
#     #genesis_block = Block(None, [], 0, None, 0, 0, 0)
#
#     # Create simulation
#     #s = Simulation(genesis_block=genesis_block, miners=miners, tx_file='data/txs', start_timestamp=0, end_timestamp=100, t=t, i=i)
#     s = Simulation(genesis_block=genesis_block, miners=miners,
#                    tx_file='/4tbssd/undercutting_project/summary_txs_2020_Feb', start_timestamp=1577835688,
#                    end_timestamp=1583019688, t=t, i=i)  # real_end_timestamp = 1581292777
#
#     last_block, num_forks = s.run()
#     result = get_miner_rewards(last_block)
#
#     f_out = open("results/nonstubborn_undercutting_1miner_40percent_honest_0miner_0percent_leaveout_{0}_iteration_{1}".format(t,i), 'w')
#     f_out.write(json.dumps(result)+"\n")
#     f_out.write("num_forks:" + str(num_forks) + "\n")
#     f_out.flush()
#     f_out.close()
#
#
#
# def setup_10honest_40_nonstubborn(t,i):
#
#     # # Create miners
#     m0 = UndercutMiner(name=0, mining_power=0.4, mempool_threshold=t)
#     m1 = HonestMiner(name=1, mining_power=0.1)
#     miners = [m0,m1]
#
#     len_miners = len(miners)
#     mining_powers = [.1,.1,.1,.1,.1]
#     for id, pow in enumerate(mining_powers):
#         miners.append(NormalMiner(name=id + len_miners , mining_power=pow))
#
#     # Create genesis block
#     genesis_block = Block(None, [], 1577835687, None, 0, 0, 0)
#     #genesis_block = Block(None, [], 0, None, 0, 0, 0)
#
#     # Create simulation
#     #s = Simulation(genesis_block=genesis_block, miners=miners, tx_file='data/txs', start_timestamp=0, end_timestamp=100, t=t, i=i)
#     s = Simulation(genesis_block=genesis_block, miners=miners,
#                    tx_file='/4tbssd/undercutting_project/summary_txs_2020_Feb', start_timestamp=1577835688,
#                    end_timestamp=1583019688, t=t, i=i)  # real_end_timestamp = 1581292777
#
#     last_block, num_forks = s.run()
#     result = get_miner_rewards(last_block)
#
#     f_out = open("results/nonstubborn_undercutting_1miner_40percent_honest_1miner_10percent_leaveout_{0}_iteration_{1}".format(t,i), 'w')
#     f_out.write(json.dumps(result)+"\n")
#     f_out.write("num_forks:" + str(num_forks) + "\n")
#     f_out.flush()
#     f_out.close()
#
#
#
# def setup_30honest_40_nonstubborn(t,i):
#
#     # # Create miners
#     m0 = UndercutMiner(name=0, mining_power=0.4, mempool_threshold=t)
#     m1 = HonestMiner(name=1, mining_power=0.1)
#     m2 = HonestMiner(name=2, mining_power=0.1)
#     m3 = HonestMiner(name=3, mining_power=0.1)
#     miners = [m0,m1,m2,m3]
#
#     len_miners = len(miners)
#     mining_powers = [.1,.1,.1]
#     for id, pow in enumerate(mining_powers):
#         miners.append(NormalMiner(name=id + len_miners , mining_power=pow))
#
#     # Create genesis block
#     genesis_block = Block(None, [], 1577835687, None, 0, 0, 0)
#     #genesis_block = Block(None, [], 0, None, 0, 0, 0)
#
#     # Create simulation
#     #s = Simulation(genesis_block=genesis_block, miners=miners, tx_file='data/txs', start_timestamp=0, end_timestamp=100, t=t, i=i)
#     s = Simulation(genesis_block=genesis_block, miners=miners,
#                    tx_file='/4tbssd/undercutting_project/summary_txs_2020_Feb', start_timestamp=1577835688,
#                    end_timestamp=1583019688, t=t, i=i)  # real_end_timestamp = 1581292777
#
#     last_block, num_forks = s.run()
#     result = get_miner_rewards(last_block)
#
#     f_out = open("results/nonstubborn_undercutting_1miner_40percent_honest_3miner_30percent_leaveout_{0}_iteration_{1}".format(t,i), 'w')
#     f_out.write(json.dumps(result)+"\n")
#     f_out.write("num_forks:" + str(num_forks) + "\n")
#     f_out.flush()
#     f_out.close()
#
#
#
# def setup_50honest_40_nonstubborn(t,i):
#
#     # # Create miners
#     m0 = UndercutMiner(name=0, mining_power=0.4, mempool_threshold=t)
#     m1 = HonestMiner(name=1, mining_power=0.1)
#     m2 = HonestMiner(name=2, mining_power=0.1)
#     m3 = HonestMiner(name=3, mining_power=0.1)
#     m4 = HonestMiner(name=4, mining_power=0.1)
#     m5 = HonestMiner(name=5, mining_power=0.1)
#     miners = [m0,m1,m2,m3,m4,m5]
#
#     len_miners = len(miners)
#     mining_powers = [.1]
#     for id, pow in enumerate(mining_powers):
#         miners.append(NormalMiner(name=id + len_miners , mining_power=pow))
#
#     # Create genesis block
#     genesis_block = Block(None, [], 1577835687, None, 0, 0, 0)
#     #genesis_block = Block(None, [], 0, None, 0, 0, 0)
#
#     # Create simulation
#     #s = Simulation(genesis_block=genesis_block, miners=miners, tx_file='data/txs', start_timestamp=0, end_timestamp=100, t=t, i=i)
#     s = Simulation(genesis_block=genesis_block, miners=miners,
#                    tx_file='/4tbssd/undercutting_project/summary_txs_2020_Feb', start_timestamp=1577835688,
#                    end_timestamp=1583019688, t=t, i=i)  # real_end_timestamp = 1581292777
#
#     last_block, num_forks = s.run()
#     result = get_miner_rewards(last_block)
#
#     f_out = open("results/nonstubborn_undercutting_1miner_40percent_honest_5miner_50percent_leaveout_{0}_iteration_{1}".format(t,i), 'w')
#     f_out.write(json.dumps(result)+"\n")
#     f_out.write("num_forks:" + str(num_forks) + "\n")
#     f_out.flush()
#     f_out.close()
#
# ##########################################################
# ##########################################################
# ##########################################################
#
# def setup_0honest_40_nonstubborn(t,i):
#
#     # # Create miners
#     m0 = UndercutMiner(name=0, mining_power=0.4, mempool_threshold=t)
#     miners = [m0]
#
#     len_miners = len(miners)
#     mining_powers = [.1,.1,.1,.1,.1,.1]
#     for id, pow in enumerate(mining_powers):
#         miners.append(NormalMiner(name=id + len_miners , mining_power=pow))
#
#     # Create genesis block
#     genesis_block = Block(None, [], 1577835687, None, 0, 0, 0)
#     #genesis_block = Block(None, [], 0, None, 0, 0, 0)
#
#     # Create simulation
#     #s = Simulation(genesis_block=genesis_block, miners=miners, tx_file='data/txs', start_timestamp=0, end_timestamp=100, t=t, i=i)
#     s = Simulation(genesis_block=genesis_block, miners=miners,
#                    tx_file='/4tbssd/undercutting_project/summary_txs_2020_Feb', start_timestamp=1577835688,
#                    end_timestamp=1583019688, t=t, i=i)  # real_end_timestamp = 1581292777
#
#     last_block, num_forks = s.run()
#     result = get_miner_rewards(last_block)
#
#     f_out = open("results/nonstubborn_undercutting_1miner_40percent_honest_0miner_0percent_leaveout_{0}_iteration_{1}".format(t,i), 'w')
#     f_out.write(json.dumps(result)+"\n")
#     f_out.write("num_forks:" + str(num_forks) + "\n")
#     f_out.flush()
#     f_out.close()
#
#
#
# def setup_10honest_40_nonstubborn(t,i):
#
#     # # Create miners
#     m0 = UndercutMiner(name=0, mining_power=0.4, mempool_threshold=t)
#     m1 = HonestMiner(name=1, mining_power=0.1)
#     miners = [m0,m1]
#
#     len_miners = len(miners)
#     mining_powers = [.1,.1,.1,.1,.1]
#     for id, pow in enumerate(mining_powers):
#         miners.append(NormalMiner(name=id + len_miners , mining_power=pow))
#
#     # Create genesis block
#     genesis_block = Block(None, [], 1577835687, None, 0, 0, 0)
#     #genesis_block = Block(None, [], 0, None, 0, 0, 0)
#
#     # Create simulation
#     #s = Simulation(genesis_block=genesis_block, miners=miners, tx_file='data/txs', start_timestamp=0, end_timestamp=100, t=t, i=i)
#     s = Simulation(genesis_block=genesis_block, miners=miners,
#                    tx_file='/4tbssd/undercutting_project/summary_txs_2020_Feb', start_timestamp=1577835688,
#                    end_timestamp=1583019688, t=t, i=i)  # real_end_timestamp = 1581292777
#
#     last_block, num_forks = s.run()
#     result = get_miner_rewards(last_block)
#
#     f_out = open("results/nonstubborn_undercutting_1miner_40percent_honest_1miner_10percent_leaveout_{0}_iteration_{1}".format(t,i), 'w')
#     f_out.write(json.dumps(result)+"\n")
#     f_out.write("num_forks:" + str(num_forks) + "\n")
#     f_out.flush()
#     f_out.close()
#
#
#
# def setup_30honest_40_nonstubborn(t,i):
#
#     # # Create miners
#     m0 = UndercutMiner(name=0, mining_power=0.4, mempool_threshold=t)
#     m1 = HonestMiner(name=1, mining_power=0.1)
#     m2 = HonestMiner(name=2, mining_power=0.1)
#     m3 = HonestMiner(name=3, mining_power=0.1)
#     miners = [m0,m1,m2,m3]
#
#     len_miners = len(miners)
#     mining_powers = [.1,.1,.1]
#     for id, pow in enumerate(mining_powers):
#         miners.append(NormalMiner(name=id + len_miners , mining_power=pow))
#
#     # Create genesis block
#     genesis_block = Block(None, [], 1577835687, None, 0, 0, 0)
#     #genesis_block = Block(None, [], 0, None, 0, 0, 0)
#
#     # Create simulation
#     #s = Simulation(genesis_block=genesis_block, miners=miners, tx_file='data/txs', start_timestamp=0, end_timestamp=100, t=t, i=i)
#     s = Simulation(genesis_block=genesis_block, miners=miners,
#                    tx_file='/4tbssd/undercutting_project/summary_txs_2020_Feb', start_timestamp=1577835688,
#                    end_timestamp=1583019688, t=t, i=i)  # real_end_timestamp = 1581292777
#
#     last_block, num_forks = s.run()
#     result = get_miner_rewards(last_block)
#
#     f_out = open("results/nonstubborn_undercutting_1miner_40percent_honest_3miner_30percent_leaveout_{0}_iteration_{1}".format(t,i), 'w')
#     f_out.write(json.dumps(result)+"\n")
#     f_out.write("num_forks:" + str(num_forks) + "\n")
#     f_out.flush()
#     f_out.close()
#
#
#
# def setup_50honest_40_nonstubborn(t,i):
#
#     # # Create miners
#     m0 = UndercutMiner(name=0, mining_power=0.4, mempool_threshold=t)
#     m1 = HonestMiner(name=1, mining_power=0.1)
#     m2 = HonestMiner(name=2, mining_power=0.1)
#     m3 = HonestMiner(name=3, mining_power=0.1)
#     m4 = HonestMiner(name=4, mining_power=0.1)
#     m5 = HonestMiner(name=5, mining_power=0.1)
#     miners = [m0,m1,m2,m3,m4,m5]
#
#     len_miners = len(miners)
#     mining_powers = [.1]
#     for id, pow in enumerate(mining_powers):
#         miners.append(NormalMiner(name=id + len_miners , mining_power=pow))
#
#     # Create genesis block
#     genesis_block = Block(None, [], 1577835687, None, 0, 0, 0)
#     #genesis_block = Block(None, [], 0, None, 0, 0, 0)
#
#     # Create simulation
#     #s = Simulation(genesis_block=genesis_block, miners=miners, tx_file='data/txs', start_timestamp=0, end_timestamp=100, t=t, i=i)
#     s = Simulation(genesis_block=genesis_block, miners=miners,
#                    tx_file='/4tbssd/undercutting_project/summary_txs_2020_Feb', start_timestamp=1577835688,
#                    end_timestamp=1583019688, t=t, i=i)  # real_end_timestamp = 1581292777
#
#     last_block, num_forks = s.run()
#     result = get_miner_rewards(last_block)
#
#     f_out = open("results/nonstubborn_undercutting_1miner_40percent_honest_5miner_50percent_leaveout_{0}_iteration_{1}".format(t,i), 'w')
#     f_out.write(json.dumps(result)+"\n")
#     f_out.write("num_forks:" + str(num_forks) + "\n")
#     f_out.flush()
#     f_out.close()
#
# ##########################################################
# ##########################################################
# ##########################################################
#
#
# def setup_0honest_45_nonstubborn(t,i):
#     print("0")
#     time.sleep(2)
#     # # Create miners
#     m0 = UndercutMiner(name=0, mining_power=0.45, mempool_threshold=t)
#     miners = [m0]
#
#     len_miners = len(miners)
#     mining_powers = [.1,.1,.1,.1,.1,.05]
#     for id, pow in enumerate(mining_powers):
#         miners.append(NormalMiner(name=id + len_miners , mining_power=pow))
#
#     # Create genesis block
#     genesis_block = Block(None, [], 1577835687, None, 0, 0, 0)
#     #genesis_block = Block(None, [], 0, None, 0, 0, 0)
#
#     # Create simulation
#     #s = Simulation(genesis_block=genesis_block, miners=miners, tx_file='data/txs', start_timestamp=0, end_timestamp=100, t=t, i=i)
#     s = Simulation(genesis_block=genesis_block, miners=miners,
#                    tx_file='/4tbssd/undercutting_project/summary_txs_2020_Feb', start_timestamp=1577835688,
#                    end_timestamp=1582019688, t=t, i=i)  # real_end_timestamp = 1581292777
#
#     last_block, num_forks = s.run()
#     result = get_miner_rewards(last_block)
#
#     f_out = open("results2/nonstubborn1_undercutting_1miner_45percent_honest_0miner_0percent_leaveout_{0}_iteration_{1}".format(t,i), 'w')
#     f_out.write(json.dumps(result)+"\n")
#     f_out.write("num_forks:" + str(num_forks) + "\n")
#     f_out.flush()
#     f_out.close()
#
#
#
# def setup_10honest_45_nonstubborn(t,i):
#     print("10")
#     time.sleep(2)
#     # # Create miners
#     m0 = UndercutMiner(name=0, mining_power=0.45, mempool_threshold=t)
#     m1 = HonestMiner(name=1, mining_power=0.1)
#     miners = [m0,m1]
#
#     len_miners = len(miners)
#     mining_powers = [.1,.1,.1,.1,.05]
#     for id, pow in enumerate(mining_powers):
#         miners.append(NormalMiner(name=id + len_miners , mining_power=pow))
#
#     # Create genesis block
#     genesis_block = Block(None, [], 1577835687, None, 0, 0, 0)
#     #genesis_block = Block(None, [], 0, None, 0, 0, 0)
#
#     # Create simulation
#     #s = Simulation(genesis_block=genesis_block, miners=miners, tx_file='data/txs', start_timestamp=0, end_timestamp=100, t=t, i=i)
#     s = Simulation(genesis_block=genesis_block, miners=miners,
#                    tx_file='/4tbssd/undercutting_project/summary_txs_2020_Feb', start_timestamp=1577835688,
#                    end_timestamp=1582019688, t=t, i=i)  # real_end_timestamp = 1581292777
#
#     last_block, num_forks = s.run()
#     result = get_miner_rewards(last_block)
#
#     f_out = open("results2/nonstubborn1_undercutting_1miner_45percent_honest_1miner_10percent_leaveout_{0}_iteration_{1}".format(t,i), 'w')
#     f_out.write(json.dumps(result)+"\n")
#     f_out.write("num_forks:" + str(num_forks) + "\n")
#     f_out.flush()
#     f_out.close()
#
#
#
# def setup_30honest_45_nonstubborn(t,i):
#     print("30")
#     time.sleep(2)
#     # # Create miners
#     m0 = UndercutMiner(name=0, mining_power=0.45, mempool_threshold=t)
#     m1 = HonestMiner(name=1, mining_power=0.1)
#     m2 = HonestMiner(name=2, mining_power=0.1)
#     m3 = HonestMiner(name=3, mining_power=0.1)
#     miners = [m0,m1,m2,m3]
#
#     len_miners = len(miners)
#     mining_powers = [.1,.1,.05]
#     for id, pow in enumerate(mining_powers):
#         miners.append(NormalMiner(name=id + len_miners , mining_power=pow))
#
#     # Create genesis block
#     genesis_block = Block(None, [], 1577835687, None, 0, 0, 0)
#     #genesis_block = Block(None, [], 0, None, 0, 0, 0)
#
#     # Create simulation
#     #s = Simulation(genesis_block=genesis_block, miners=miners, tx_file='data/txs', start_timestamp=0, end_timestamp=100, t=t, i=i)
#     s = Simulation(genesis_block=genesis_block, miners=miners,
#                    tx_file='/4tbssd/undercutting_project/summary_txs_2020_Feb', start_timestamp=1577835688,
#                    end_timestamp=1582019688, t=t, i=i)  # real_end_timestamp = 1581292777
#
#     last_block, num_forks = s.run()
#     result = get_miner_rewards(last_block)
#
#     f_out = open("results2/nonstubborn1_undercutting_1miner_45percent_honest_3miner_30percent_leaveout_{0}_iteration_{1}".format(t,i), 'w')
#     f_out.write(json.dumps(result)+"\n")
#     f_out.write("num_forks:" + str(num_forks) + "\n")
#     f_out.flush()
#     f_out.close()
#
#
#
# def setup_50honest_45_nonstubborn(t,i):
#     print("50")
#     time.sleep(2)
#     # # Create miners
#     m0 = UndercutMiner(name=0, mining_power=0.45, mempool_threshold=t)
#     m1 = HonestMiner(name=1, mining_power=0.1)
#     m2 = HonestMiner(name=2, mining_power=0.1)
#     m3 = HonestMiner(name=3, mining_power=0.1)
#     m4 = HonestMiner(name=4, mining_power=0.1)
#     m5 = HonestMiner(name=5, mining_power=0.1)
#     miners = [m0,m1,m2,m3,m4,m5]
#
#     len_miners = len(miners)
#     mining_powers = [.05]
#     for id, pow in enumerate(mining_powers):
#         miners.append(NormalMiner(name=id + len_miners , mining_power=pow))
#
#     # Create genesis block
#     genesis_block = Block(None, [], 1577835687, None, 0, 0, 0)
#     #genesis_block = Block(None, [], 0, None, 0, 0, 0)
#
#     # Create simulation
#     #s = Simulation(genesis_block=genesis_block, miners=miners, tx_file='data/txs', start_timestamp=0, end_timestamp=100, t=t, i=i)
#     s = Simulation(genesis_block=genesis_block, miners=miners,
#                    tx_file='/4tbssd/undercutting_project/summary_txs_2020_Feb', start_timestamp=1577835688,
#                    end_timestamp=1582019688, t=t, i=i)  # real_end_timestamp = 1581292777
#
#     last_block, num_forks = s.run()
#     result = get_miner_rewards(last_block)
#
#     f_out = open("results2/nonstubborn1_undercutting_1miner_45percent_honest_5miner_50percent_leaveout_{0}_iteration_{1}".format(t,i), 'w')
#     f_out.write(json.dumps(result)+"\n")
#     f_out.write("num_forks:" + str(num_forks) + "\n")
#     f_out.flush()
#     f_out.close()
#
# ##########################################################
# ##########################################################
# ##########################################################
