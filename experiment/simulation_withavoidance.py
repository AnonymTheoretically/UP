import time
from scipy.stats import geom
import math
import numpy as np
import scipy.integrate as integrate
import json
from chain import Chain
from block import Block
from mempool import Mempool
from miner import Miner, NormalMiner, UndercutMiner, HonestMiner
from tx import Tx
import random
from multiprocessing import Pool
import time
import itertools
import gc
import os.path
import statistics
import logging, sys
logger = logging.getLogger()
logger.setLevel(logging.ERROR)
#logger.setLevel(logging.DEBUG)




avg_block_time = 600
block_size = 1266149
block_size = 1183248
block_size = 1272938
block_size = 1261269  # TIAN JULY
block_size = 1273706  # TIAN JUNE
block_size = 1000*1000  # FAKE DATA
#block_size = 2000000
inflation_factor = 1
#avoidance_rate = 0.98
avoidance_rate = 0.9

ignore_first_blocks = 10
ignore_last_blocks = 10
negligible = 0.01
begining_timestamp = 1589540396  #JUNE
begining_timestamp = 1586909337  #JULY
ending_timestamp = 1594856530   #july
ending_timestamp = 1592265346   #june
begining_timestamp = 1592265804  #TIAN JULY
ending_timestamp = 1594856530   #TIAN july
begining_timestamp = 1589540396  #TIAN JUNE
ending_timestamp = 1592265346   #TIAN JUNE

begining_timestamp = 0      #FAKE DATA
ending_timestamp = 6000000   #FAKE DATA
deadline = 1600000000
stop_block = 1000

tx_file_path = "/home/mohsen/get_new_data_undercutting/summary_txs_2020_JuneJuly"
tx_file_path = "/home/mohsen/get_new_data_undercutting/summary_txs_2020_June"
tx_file_path = "/home/mohsen/get_new_data_undercutting/summary_txs_2020_July"
tx_file_path2 = "/home/mohsen/get_new_data_undercutting/tian_data_June15_to_July15"
tx_file_path1 = "/home/mohsen/get_new_data_undercutting/tian_data_May15_to_June15"
tx_file_path1 = "/home/mohsen/get_new_data_undercutting/fake_bicoin_3M_txs_normal" ##FAKE DATA

result_path = "results/blocksize_1mb/"
result_path = "results/blocksize_1mb_2months/"
result_path = "results/bitcoin_1month_0.98avoidance_1inflation/"
result_path = "results/test/july_0.9/"
result_path = "results/test/june_0.9/"
result_path = "results/test/june_0.8/"
result_path = "results/test/june_0.98/"
result_path = "results/bitcoin_1month_0.9avoidance_1inflation/"
result_path = "results/bitcoin_2month_0.9avoidance_1inflation/"
result_path = "results/bitcoin_1month_0.9avoidance_1.2inflation/"
result_path = "results/bitcoin_1month_0.9avoidance_1inflation/"
result_path = "results/bitcoin_0.5month_0.8avoidance_1inflation/"
result_path = "results/bitcoin_1month_1avoidance_1inflation/"
result_path = "results/bitcoin_1month_0.8avoidance_1inflation/"
result_path = "results/bitcoin_0.5month_0.8avoidance_1inflation/"
#result_path = "results/bitcoin_0.5month_0.8avoidance_1inflation_future/"
#result_path = "results/bitcoin_0.5month_0.8avoidance_0.8inflation_future/"
result_path = "results/bitcoin_0.5month_0.9avoidance_1inflation/"
result_path = "results/fake_bitcoin_0.9avoidance_normal/" ##FAKE DATA


global all_txs
all_txs = {}
def load_txs(tx_file_):
    #all_txs = {}
    serial = 0
    #for tx_file in [tx_file_path1, tx_file_path2]:
    for tx_file in [tx_file_path1]:
        for id, l in enumerate(open(tx_file)):
            print(serial)
            if serial ==  5000000:
                return tx['timestamp']
            tx = json.loads(l)
            tx['serial'] = serial
            try:
                all_txs[tx['timestamp']].append(tx)
            except:
                all_txs[tx['timestamp']] = [tx]
            serial += 1
    return tx['timestamp']




# avg_block_time = 120
# block_size = 300000
# inflation_factor = 1.
# avoidance_rate = 0.8
# ignore_first_blocks = 10
# ignore_last_blocks = 10
# negligible = 0.01
# begining_timestamp = 1589683808
# ending_timestamp = 1600627133
# tx_file_path = "/home/mohsen/get_new_data_undercutting/monero_txs_block2100000To2191000"
# result_path = "results/monero_0.8avoidance_1inflation/"
# deadline = 1600927133
#
# global all_txs
# all_txs = {}
# def load_txs(tx_file):
#     counter = 0
#     for id, l in enumerate(open(tx_file)):
#         print(id)
#         tx = json.loads(l)
#         try:
#             tx['timestamp'] = tx['tx_time']
#             tx['fee'] = tx['tx_fee']
#             tx['size'] = tx['tx_size']
#             tx['serial'] = counter
#             counter += 1
#         except:
#             continue
#         try:
#             all_txs[tx['timestamp']].append(tx)
#         except:
#             all_txs[tx['timestamp']] = [tx]
#     return tx['timestamp']
#


# avg_block_time = 5
# block_size = 100
# begining_timestamp = 0s
# result_path = "data/"
# tx_file_path = "data/sample1"


class Simulation():
    def __init__(self, genesis_block, miners, secondUndercutter, start_timestamp, end_timestamp, cutoff, undercutter_power, honest_power=0, avg_block_time = avg_block_time, avoidance=False):
        #end_timestamp, self.all_txs = load_txs(tx_file_path)
        self.block_times = []
        self.txs_set = set()
        self.block_size = block_size
        self.avoidance = avoidance
        self.undercutter_power = undercutter_power
        self.honest_power = honest_power
        self.rational_power = 1 - undercutter_power - honest_power
        self.second_undercutter_power = secondUndercutter

        self.conf_blocks = 6
        self.avg_block_time = avg_block_time
        self.start_time = int(start_timestamp)
        self.current_time = int(start_timestamp)
        self.end_time = int(end_timestamp)
        self.experiment_end_flag = False

        self.D = cutoff
        self.gamma = 0

        fork_conditions = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0}
        self.fork_status ={"forks": 0, "failed_forks": 0 , "fork_len": [0], "main_len":[0], "conditions": fork_conditions, "rational_changing_to_fork": 0, "rational_changing_to_main": 0}
        self.avoidance_status ={}

        self.genesis_block = genesis_block
        self.chains = [Chain(miners, genesis_block)]
        self.chains[-1].next_block_time = self.current_time
        self.chains[-1].prev_time = self.current_time-1
        self.miners = miners
        for miner in self.miners:
            if isinstance(miner, UndercutMiner):  # considering only one undercutter, for multiple change to list or map
                self.undercutter = miner

        self.mempool_stat = {'size':[], 'fees': []}

    def run(self):
        next_extending_chain = self.get_extending_chain()  # get the extending chain for this round
        next_block_miner = np.random.choice(next_extending_chain.miners, p=next_extending_chain.get_mining_weights())  # miner that mines the block
        num_blocks = 0

        while True:
            num_blocks += 1
            #if int((self.current_time - self.start_time)/3600) % 20 == 0:
            print("{0}_{1}_{2}_{3}".format(self.undercutter_power, self.honest_power, self.avoidance, avoidance_rate) + ":   ", int((self.current_time - self.start_time)/3600))

            t0 = time.time()
            extending_chain = next_extending_chain
            miner = next_block_miner
            #extending_chain = self.get_extending_chain()  # get the extending chain for this round
            self.current_time = extending_chain.next_block_time  # update current time to the time of the block creation
            logging.debug("extending chain: {0}".format(extending_chain.get_block(-1).height))
            print("*******************************extending chain: {0}".format(extending_chain.get_block(-1).height))
            print("*******************************     ",len(extending_chain.mempool.txs))
            print("*******************************     ",extending_chain.mempool.size/1000000)

            #t0 = time.time()
            #miner = np.random.choice(extending_chain.miners, p=extending_chain.get_mining_weights())  # miner that mines the block
            logging.debug("miner selection: {0}".format(time.time() - t0))

            # t0 = time.time()
            self.publish_block(extending_chain, miner)
            logging.debug("publish block: {0}".format(time.time() - t0))

            #t0 = time.time()
            self.update_mempool(extending_chain)
            #self.update_mempool(extending_chain.prev_time, self.current_time, new_block)
            logging.debug("mempool update: {0}".format(time.time() - t0))

            # t0 = time.time()
            self.update_miners_chain(extending_chain)
            logging.debug("miner chain update: {0}".format(time.time() - t0))

            # t0 = time.time()
            self.remove_forked_chains(extending_chain)
            logging.debug("remove fork: {0}".format(time.time() - t0))

            # t0 = time.time()
            if self.honest_power < 1 and miner != self.undercutter:
                self.fork(extending_chain)
            logging.debug("add a fork: {0}".format(time.time() - t0))


            next_extending_chain = self.get_extending_chain()  # get the next extending chain
            next_block_miner = np.random.choice(next_extending_chain.miners, p=next_extending_chain.get_mining_weights())  # miner that mines the block
            #t0 = time.time()
            self.set_next_block(next_extending_chain, next_block_miner)
            logging.debug("set next block: {0}".format(time.time() - t0))



            #input("--------------")
            print("--------------")

            self.mempool_stat['size'].append(extending_chain.mempool.size/1000000)
            self.mempool_stat['fees'].append(extending_chain.mempool.fees//1000000)


            #if (self.experiment_end_flag and extending_chain.mempool.size < 10 * block_size) or self.current_time > deadline :
            #if extending_chain.get_block(-1).height > stop_block :
            if self.experiment_end_flag :
                break


        return self.get_miner_rewards(extending_chain.get_block(-1), num_blocks)

# ===================================================================================================================
    def publish_block(self, extending_chain, miner):
        t0 = time.time()
        extending_chain.publish_block(miner, self.current_time)
        logging.debug("                Publishing: ", time.time()-t0)
        mining_time = geom.rvs(p=extending_chain.get_mining_power_sum() / self.avg_block_time)
        extending_chain.next_block_time = self.current_time + mining_time
        self.block_times.append(mining_time)

# ===================================================================================================================
    def set_next_block(self, extending_chain, next_block_miner):
        t0 = time.time()
        if len(extending_chain.blocks) > 1:
            self.avoidance_check(extending_chain, next_block_miner)
        else: # this is the case that the undercutter created a new fork and has already done the avoidance check in fork()
            extending_chain.avoidance_portion = 1

        try:
            self.avoidance_status[str(extending_chain.avoidance_condition)] += 1
        except:
            self.avoidance_status[str(extending_chain.avoidance_condition)] = 1

        logging.debug("avoidance check: {0}".format(time.time() - t0))


        extending_chain.set_next_block(block_size)
        return 1

#===================================================================================================================
    def get_extending_chain(self): # find the next chain to extend based on the earliest next block time.
        min_time = math.inf
        extending_chain = self.chains[0]
        for chain in self.chains:
            if chain.next_block_time < min_time:
                min_time = chain.next_block_time
                extending_chain = chain
        return extending_chain

# ===================================================================================================================
    def avoidance_check(self, extending_chain, next_block_miner):
        if not self.avoidance:
            extending_chain.avoidance_condition = 0
            extending_chain.avoidance_portion = 1
            return 1

        bandwidth_set_fees = extending_chain.mempool.claimable_fees(num_blocks=1, block_size=block_size)
        next_bandwidth_set_fees = extending_chain.mempool.claimable_fees(num_blocks=2, block_size=block_size) - bandwidth_set_fees
        try:
            gamma =  next_bandwidth_set_fees / bandwidth_set_fees
        except:
            extending_chain.avoidance_condition = 1
            extending_chain.avoidance_portion = 0
            return 1

        Bh = self.honest_power
        Bu = self.undercutter_power
        Br = 1 - Bh - Bu


        if isinstance(extending_chain.get_block(-1).miner, NormalMiner):
            Bh += extending_chain.get_block(-1).miner.mining_power
            Br -= extending_chain.get_block(-1).miner.mining_power

        if isinstance(next_block_miner, UndercutMiner):
            Bu = self.second_undercutter_power
            Br = Br - self.second_undercutter_power + self.undercutter_power


        if self.D == 1:
            T = (Bh*Br + Bu*(1 - Bu)) / (1 - Bu)**2

            if gamma < T:
                extending_chain.avoidance_condition = 2
                extending_chain.avoidance_portion = (1+gamma)/(1+T) * avoidance_rate - negligible
            else:
                extending_chain.avoidance_condition = 3
                extending_chain.avoidance_portion = 1

        if self.D == 2:
            if gamma < (Bu**2)/(2*(1-Bu)):
                T = (Bu**2)/(2*(1-Bu))
            #elif (3*Bu + 2*Br + Bu*(Bu+Br)) / (8-(5*Bu + 2*Br + 3*Bu*(Bu+Br))) > gamma > ((4*Bh**2)/(Bh + Br) - Bh + 4*Bu - 2) / (3*Bh - 8*Bu + 2):
            elif (4*Bu + 2*Br - Bu*Bh) / (8*Bh + 6*Br + 3*Bu*Bh) > gamma > ((4*Bh**2)/(Bh + Br) - Bh + 4*Bu - 2) / (3*Bh - 8*Bu + 2):
                T = (4*Bu + 2*Br - Bu*Bh) / (8*Bh + 6*Br + 3*Bu*Bh)
            else:
                extending_chain.avoidance_condition = 4
                extending_chain.avoidance_portion = 1
                return 1

            if gamma < T:
                extending_chain.avoidance_condition = 5
                extending_chain.avoidance_portion = (1+gamma)/(1+T) * avoidance_rate - negligible
            else:
                extending_chain.avoidance_condition = 6
                extending_chain.avoidance_portion = 1

        return 1


# ===================================================================================================================
    def fork(self, extending_chain):
        for miner in extending_chain.miners:
            if isinstance(miner, UndercutMiner) and not miner.infork: # we fork only if the miner is of type undercutter and is not actively in a fork
                new_block = extending_chain.get_block(-1)
                if new_block.fees == 0:
                    return


                new_tx = []
                for t in range(int(self.current_time), int(extending_chain.next_block_time)):
                    try:
                        for tx in all_txs[t]:
                            new_tx.append(
                                Tx(timestamp=tx['timestamp'], size=tx['size'], fee=tx['fee'], serial=tx['serial']))
                    except:
                        pass

                #future_mempool = Mempool()
                #future_mempool.add_txs(new_tx + extending_chain.mempool.txs)
                #gamma = future_mempool.claimable_fees(num_blocks=1, block_size=block_size) / new_block.fees

                gamma = extending_chain.mempool.claimable_fees(num_blocks=1, block_size=block_size) / new_block.fees


                selected_txs = []
                fees = 0
                size = 0

                Bh = self.honest_power
                Bu = self.undercutter_power
                if isinstance(new_block.miner, NormalMiner):
                    Bh += new_block.miner.mining_power
                Br = 1 - Bh - Bu


                if self.D == 1:
                    if gamma < Bu / (1 - Bu) * inflation_factor:
                        fork_condition = 1
                        # if not self.avoidance:
                        #     selected_txs = extending_chain.get_block(-1).txs
                        #     fees = extending_chain.get_block(-1).fees
                        #     size = extending_chain.get_block(-1).size
                        # else:
                        T = self.second_undercutter_power / (1 - self.second_undercutter_power)
                        a = (1 + gamma) / (1 + T)
                        main_chain_head_fees = extending_chain.get_block(-1).fees
                        for tx in extending_chain.get_block(-1).txs[::-1]:
                            if size + tx.size > block_size:
                                break
                            if fees + tx.fee >=  a * main_chain_head_fees:
                                break
                            selected_txs.append(tx)
                            fees += tx.fee
                            size += tx.size

                    elif gamma < ((Bh*Br + Bu*(1-Bu)) / (1 - Bu)**2) * inflation_factor:
                        fork_condition = 2
                        # if not self.avoidance:
                        #     a = Bh/(Br + Bh)
                        # else:
                        #Br2 = Br - self.second_undercutter_power + Bu
                        Br2 = Br - self.second_undercutter_power
                        Bu2 = self.second_undercutter_power
                        Bh2 = Bh + Bu
                        T = (Bh2*Br2 + Bu2*(1-Bu2))/(1 - Bu2)
                        a = min((1 + gamma)/(1 + T), Bh/(Br + Bh)) - negligible

                        main_chain_head_fees = extending_chain.get_block(-1).fees
                        for tx in extending_chain.get_block(-1).txs[::-1]:
                            if size + tx.size > block_size:
                                break
                            if fees + tx.fee >= a * main_chain_head_fees:
                                break
                            selected_txs.append(tx)
                            fees += tx.fee
                            size += tx.size
                    else:
                        return


                if self.D == 2:
                    if gamma < (Bu ** 2) / (2 * (1 - Bu)) * inflation_factor:
                        fork_condition = 3
                        T = self.second_undercutter_power**2 / (2*(1 - self.second_undercutter_power))
                        a = (1 + gamma) / (1 + T)
                        main_chain_head_fees = extending_chain.get_block(-1).fees
                        for tx in extending_chain.get_block(-1).txs[::-1]:
                            if size + tx.size > block_size:
                                break
                            if fees + tx.fee >=  a * main_chain_head_fees:
                                break
                            selected_txs.append(tx)
                            fees += tx.fee
                            size += tx.size

                    elif Bh < (8*Bu - 2)/3:
                        #if (3 * Bu + 2 * Br + Bu * (Bu + Br)) / (8 - (5 * Bu + 2 * Br + 3 * Bu * (Bu + Br))) > gamma > ((4 * Bh ** 2) / (Bh + Br) - Bh + 4 * Bu - 2) / (3 * Bh - 8 * Bu + 2):
                        if (4*Bu + 2*Br - Bu*Bh) / (8*Bh + 6*Br + 3*Bu*Bh) > gamma > ((4 * (Bh ** 2)) / (Bh + Br) - Bh + 4 * Bu - 2) / (3 * Bh - 8 * Bu + 2):
                            fork_condition = 4
                            Br2 = Br - self.second_undercutter_power
                            Bu2 = self.second_undercutter_power
                            Bh2 = Bh + Bu
                            #Br2 = Br - self.second_undercutter_power + Bu
                            #Bu2 = self.second_undercutter_power

                            T = (4*Bu2 + 2*Br2 - Bu2*Bh2) / (8*Bh2 + 6*Br2 + 3*Bu2*Bh2)
                            a = min((1 + gamma)/(1 + T), Bh2/(Br + Bh2)) - negligible

                            main_chain_head_fees = extending_chain.get_block(-1).fees
                            for tx in extending_chain.get_block(-1).txs[::-1]:
                                if size + tx.size > block_size:
                                    break
                                if fees + tx.fee >= a * main_chain_head_fees:
                                    break
                                selected_txs.append(tx)
                                fees += tx.fee
                                size += tx.size
                        else:
                            return

                    else:
                        return



                #
                #
                # elif self.D == 2:
                #     if gamma < negligible:
                #         fork_condition = 4  # gamma negligible, then undercut with the first block being 1/3 of the main chain head
                #         for tx in new_block.txs:
                #             if fees + tx.fee > 0.33 * new_block.fees:
                #                 break
                #             selected_txs.append(tx)
                #             fees += tx.fee
                #             size += tx.size
                #     elif extending_chain.mempool.size < block_size:
                #         fork_condition = 5  # undercut with the first block being 0.5 current bandwidth set
                #         for tx in extending_chain.mempool.txs:
                #             if size + tx.size > block_size or fees + tx.fee > 0.5 * extending_chain.mempool.claimable_fees(
                #                     num_blocks=1, block_size=block_size):
                #                 break
                #             selected_txs.append(tx)
                #             fees += tx.fee
                #             size += tx.size
                #     elif gamma < pow(Bu, 2) / (2 - 4 * Bu + 2 * pow(Bu, 2)) * inflation_factor:
                #         fork_condition = 6  # undercut with the first block being the current bandwidth set, leaving everything in main chain head unclaimed
                #         for tx in extending_chain.mempool.txs:
                #             if size + tx.size > block_size:
                #                 break
                #             selected_txs.append(tx)
                #             fees += tx.fee
                #             size += tx.size
                #     elif gamma < min(
                #                     (pow(Bh, 2) / (1 - Bu) + Bu - Bh) / (1 - 2 * Bu),
                #                     (Bu * (1 - Bh)) / (1 + Bh - Bu)) * inflation_factor:
                #         fork_condition = 7  # undercut with the first block being the current bandwidth set, leaving everything in main chain head unclaimed
                #         for tx in extending_chain.mempool.txs:
                #             if size + tx.size > block_size:
                #                 break
                #             selected_txs.append(tx)
                #             fees += tx.fee
                #             size += tx.size
                #     else:
                #         return
                # else:
                #     return

                try:
                    self.fork_status["conditions"][str(fork_condition)] += 1
                except:
                    self.fork_status["conditions"][str(fork_condition)] = 1
                print("forking under condition: ", fork_condition)

                forked_chain = Chain(miners=[miner], origin_block=extending_chain.get_block(-2))
                forked_chain.next_block_time =  self.current_time + geom.rvs(p=forked_chain.get_mining_power_sum()/self.avg_block_time)
                forked_chain.prev_time = self.current_time
                forked_chain_block = Block(miner=miner, txs=selected_txs, timestamp=None,
                                         parent_block=extending_chain.get_block(-2), txfees=fees, size=size,
                                         height=extending_chain.get_block(-2).height+1)
                forked_chain.next_block = forked_chain_block
                forked_chain.set_mempool(extending_chain)
                self.chains.append(forked_chain)
                miner.infork = True
                self.fork_status["forks"] += 1

                self.update_next_block_time(extending_chain, miner, remove=True)   # updating the miner list in main chain and increasing the block creation time on main chain.

                print(forked_chain.next_block_time - extending_chain.next_block_time)

# ===================================================================================================================
    def update_next_block_time(self, extending_chain, miner, remove=True):
        prev_extending_chain_mining_power = extending_chain.get_mining_power_sum()
        prev_extending_chain_next_block_time = extending_chain.next_block_time
        if remove:
            extending_chain.miners.remove(miner)  # remove the miner from the miner list of prev chain
        else:
            extending_chain.miners.append(miner)
        try:
            extending_chain.next_block_time = (prev_extending_chain_next_block_time - self.current_time) * (prev_extending_chain_mining_power / extending_chain.get_mining_power_sum()) + self.current_time
        except:
            extending_chain.next_block_time = math.inf
        extending_chain.update_mining_power()
# ===================================================================================================================
    def remove_forked_chains(self, extended_chain):
        chain1 = extended_chain
        for chain2 in self.chains:
            if chain1 == chain2:
                continue
            # --------------------Find The Main and Fork Chain---------------------------------
            if self.undercutter in chain1.miners:  # check to see which chain is the forked chain by the undercutter
                fork_chain = chain1
                main_chain = chain2
            else:
                fork_chain = chain2
                main_chain = chain1
            forking_block = fork_chain.origin_block


            height_diff = extended_chain.get_block(-1).height - chain2.get_block(-1).height
            if height_diff >= self.D or len(chain2.miners) == 0:
                print(height_diff, "\n")
                for miner in chain2.miners:
                    chain1.miners.append(miner)  # add the miner of c2 to c1
                    if isinstance(miner, UndercutMiner):
                        self.fork_status["failed_forks"] += 1

                for miner in chain1.miners:  # reset the undercutting miners in the main chain
                    if isinstance(miner, UndercutMiner):
                        miner.infork = False
                chain1.update_mining_power()

                if extended_chain == main_chain:
                    self.fork_status["fork_len"].append(fork_chain.get_block(-1).height - forking_block.height)
                elif extended_chain == fork_chain:
                    self.fork_status["main_len"].append(main_chain.get_block(-1).height - forking_block.height)

                self.chains.remove(chain2)
                gc.collect()



# ===================================================================================================================
    def update_mempool(self, extending_chain):
        prev_time = extending_chain.prev_time
        curr_time = self.current_time

        t0 = time.time()
        txs = self.get_mempool_tx(prev_time, curr_time)  # get new txs for mempool
        #print("t10_1: ", time.time() - t0)

        t0 = time.time()
        for chain in self.chains:
            chain.mempool.add_txs(txs)
            chain.prev_time = self.current_time
        #print("t10_3: ", time.time() - t0)

        #extending_chain.prev_time = self.current_time


        #
        # t0 = time.time()
        # new_block.remove_txs_mempool(new_block.txs)
        # # print("t10_2: ", time.time() - t0)



    def get_mempool_tx(self, prev_time, curr_time):
        new_tx = []
        for t in range(int(prev_time), int(curr_time)):
            try:
                for tx in all_txs[t]:
                    if tx['serial'] in self.txs_set:
                        input("found a duplicate")
                        break
                    self.txs_set.add(tx['serial'])
                    new_tx.append(Tx(timestamp=tx['timestamp'], size=tx['size'], fee=tx['fee'], serial=tx['serial']))
                    if  tx['timestamp'] >= self.end_time:
                        self.experiment_end_flag = True

            except:
                pass
        return new_tx


# ===================================================================================================================
    def update_chain(self, chain, block, curr_time):
        try:
            chain.get_block(-4).mempool.txs = []
            chain.get_block(-4).txs = []
        except:
            "out of bounds"
        chain.get_block(-1).next_block = block
        #chain.add_block(block) #add the block to the chain
        chain.next_block_time = self.current_time + geom.rvs(p=chain.get_mining_power_sum()/self.avg_block_time)


 # ===================================================================================================================
    def change_chains(self, miner_change_chain):
        for miner, (fromchain, tochain) in miner_change_chain.items():
            self.update_next_block_time(fromchain, miner, remove=True)
            self.update_next_block_time(tochain, miner, remove=False)


    def update_miners_chain(self, extending_chain):
        chain1 = extending_chain #chain 1 is the current chain that is extending
        for chain2 in self.chains:
            if chain1 == chain2:
                continue
            else:
                otherChain = chain2


            # --------------------Find The Main and Fork Chain---------------------------------
            if self.undercutter in chain1.miners:  # check to see which chain is the forked chain by the undercutter
                fork_chain = chain1
                main_chain = chain2
            else:
                fork_chain = chain2
                main_chain = chain1
            height_diff = fork_chain.get_block(-1).height - main_chain.get_block(-1).height  # D~
            forking_block = fork_chain.origin_block

            if extending_chain == fork_chain:
                print("fork chain extended")
            else:
                print("main chain extended")
            print("miners:  ", extending_chain.miners)
            #--------------------Change Honest Miners' Chain---------------------------------
            if chain1.get_block(-1).height >  chain2.get_block(-1).height:  # extending chain is longer, change honest miners' chain
                miner_change_chain = {}
                for miner in chain2.miners:
                    if isinstance(miner, HonestMiner):
                        miner_change_chain[miner] = (chain2, chain1)
                        print("honest changing chains")
                if len(miner_change_chain) > 0:
                    self.change_chains(miner_change_chain)

            #---------------------Change Rational Miners' Chain for D=1-------------------------
            claimable_fees_main = main_chain.mempool.claimable_fees(self.D + height_diff, block_size=block_size)
            claimable_fees_fork = fork_chain.mempool.claimable_fees(self.D - height_diff, block_size=block_size)

            Br = self.rational_power
            Bh = self.honest_power
            Bu = self.undercutter_power
            for miner in otherChain.miners:
                if isinstance(miner, NormalMiner):
                    if miner == otherChain.get_block(-1).miner:
                        if extending_chain.get_block(-1).height > otherChain.get_block(-1).height:
                            Br -= miner.mining_power
                            Bh += miner.mining_power
                        break

            miner_change_chain = {}
            for miner in otherChain.miners:
                if isinstance(miner, NormalMiner):
                    if miner == otherChain.get_block(-1).miner:
                        if extending_chain.get_block(-1).height > otherChain.get_block(-1).height:
                            self.fork_status["rational_changing_to_fork"] += 1
                            miner_change_chain[miner] = (otherChain, extending_chain)
                        continue


                    main_profits = 0  # get the miners profits on the main chain
                    for i in range(1, main_chain.get_block(-1).height - forking_block.height + 1):
                        b = main_chain.get_block(-i)
                        if b.miner == miner:
                            main_profits += b.fees
                    fork_profits = 0  # get the miners profits on the fork chain
                    for i in range(1, fork_chain.get_block(-1).height - forking_block.height + 1):
                        b = fork_chain.get_block(-i)
                        if b.miner == miner:
                            fork_profits += b.fees

                    ##############
                    if extending_chain == fork_chain:  # fork chain extending by 1

                        Br = main_chain.get_rational_mining_power_without_block()

                        pm = pow(1 - fork_chain.get_mining_power_sum(), self.D + height_diff)
                        pf = pow(fork_chain.get_mining_power_sum(), self.D - height_diff)
                        A = main_profits*pm + fork_profits*pf + \
                            claimable_fees_main * (miner.mining_power / (1 - fork_chain.get_mining_power_sum()) * pm)
                        #claimable_fees_fork

                        pm = pow(1 - fork_chain.get_mining_power_sum() - Br, self.D + height_diff)
                        pf = pow(fork_chain.get_mining_power_sum() + Br, self.D - height_diff)
                        B = main_profits * pm + fork_profits * pf + \
                            claimable_fees_fork * (miner.mining_power / (fork_chain.get_mining_power_sum() + Br) * pf)


                    elif extending_chain == main_chain:  # fork chain extending by 1

                        Br = fork_chain.get_rational_mining_power_without_block()

                        pm = pow(1 - fork_chain.get_mining_power_sum(), self.D + height_diff)
                        pf = pow(fork_chain.get_mining_power_sum(), self.D - height_diff)
                        A = main_profits * pm + fork_profits * pf + \
                            claimable_fees_fork * (miner.mining_power / fork_chain.get_mining_power_sum() * pf)
                        #

                        pm = pow(1 - fork_chain.get_mining_power_sum() + Br, self.D + height_diff)
                        pf = pow(fork_chain.get_mining_power_sum() - Br, self.D - height_diff)
                        B = main_profits * pm + fork_profits * pf + \
                            claimable_fees_main * \
                            (miner.mining_power / (1 - fork_chain.get_mining_power_sum() + Br) * pm)

                    if A < B:
                        miner_change_chain[miner] = (chain2, chain1)

            if len(miner_change_chain) > 0:
                self.change_chains(miner_change_chain)



#          #---------------------Change Rational Miners' Chain for D=2-------------------------
# #           if self.D == 2:
#             miner_change_chain = {}
#             claimable_fees_main = main_chain.mempool.claimable_fees(self.D + height_diff, block_size=block_size)
#             claimable_fees_fork = fork_chain.mempool.claimable_fees(self.D - height_diff, block_size=block_size)
#
#             if extending_chain == fork_chain: # fork chain extending by 1
#                 Br = main_chain.get_rational_mining_power()
#
#                 for miner in main_chain.miners:
#                     if isinstance(miner, NormalMiner):
#                         if miner == main_chain.get_block(-1).miner:
#                             if extending_chain.get_block(-1).height > chain2.get_block(-1).height:
#                                 self.fork_status["rational_changing_to_fork"] += 1
#                                 miner_change_chain[miner] = (chain2, extending_chain)
#                                 #self.change_chains(miner_change_chain)
#                                 continue
#                         main_profits = 0  # get the miners profits on the main chain
#                         for i in range(1, main_chain.get_block(-1).height - forking_block.height + 1):
#                             b = main_chain.get_block(-i)
#                             if b.miner == miner:
#                                 main_profits += b.fees
#                         fork_profits = 0  # get the miners profits on the fork chain
#                         for i in range(1, fork_chain.get_block(-1).height - forking_block.height + 1):
#                             b = fork_chain.get_block(-i)
#                             if b.miner == miner:
#                                 fork_profits += b.fees
#
#                         if isinstance(fork_chain.get_block(-1).miner, NormalMiner):
#                             Br -= fork_chain.get_block(-1).miner.mining_power
#
#                         pm = pow(1 - fork_chain.get_mining_power_sum(), self.D + height_diff)
#                         pf = pow(fork_chain.get_mining_power_sum(), self.D - height_diff)
#                         A = main_profits*pm + fork_profits*pf + claimable_fees_main * (miner.mining_power/(1-fork_chain.get_mining_power_sum())*pm)
#                         #
#                         pm = pow(1 - fork_chain.get_mining_power_sum() - Br, self.D + height_diff)
#                         pf = pow(fork_chain.get_mining_power_sum() + Br, self.D - height_diff)
#                         B = main_profits*pm + fork_profits*pf + claimable_fees_fork * (miner.mining_power/(fork_chain.get_mining_power_sum()+Br)*pf)
#
#                         if A < B:
#                             miner_change_chain[miner] = (chain2, chain1)
#                 if len(miner_change_chain) > 0:
#                     self.fork_status["rational_changing_to_fork"] += 1
#                     self.change_chains(miner_change_chain)
#
#             elif extending_chain == main_chain: # main chain extending by 1
#                 Br = fork_chain.get_rational_mining_power()
#
#                 for miner in fork_chain.miners:
#                     if isinstance(miner, NormalMiner):
#
#                         if miner == fork_chain.get_block(-1).miner:
#                             if extending_chain.get_block(-1).height > chain2.get_block(-1).height:
#                                 self.fork_status["rational_changing_to_main"] += 1
#                                 miner_change_chain[miner] = (chain2, extending_chain)
#                                 #self.change_chains(miner_change_chain)
#                                 continue
#
#                         main_profits = 0  # get the miners profits on the main chain
#                         for i in range(1, main_chain.get_block(-1).height - forking_block.height + 1):
#                             b = main_chain.get_block(-i)
#                             if b.miner == miner:
#                                 main_profits += b.fees
#                         fork_profits = 0  # get the miners profits on the fork chain
#                         for i in range(1, fork_chain.get_block(-1).height - forking_block.height + 1):
#                             b = fork_chain.get_block(-i)
#                             if b.miner == miner:
#                                 fork_profits += b.fees
#
#                         if isinstance(main_chain.get_block(-1).miner, NormalMiner):
#                             Br -= main_chain.get_block(-1).miner.mining_power
#
#                         pm = pow(1 - fork_chain.get_mining_power_sum(), self.D + height_diff)
#                         pf = pow(fork_chain.get_mining_power_sum(), self.D - height_diff)
#                         A = main_profits*pm + fork_profits*pf + claimable_fees_fork * (miner.mining_power / fork_chain.get_mining_power_sum() * pm)
#                         #
#                         pm = pow(1 - fork_chain.get_mining_power_sum() + Br, self.D + height_diff)
#                         pf = pow(fork_chain.get_mining_power_sum() - Br, self.D - height_diff)
#                         B = main_profits * pm + fork_profits * pf + claimable_fees_main * (miner.mining_power / (1 - fork_chain.get_mining_power_sum() + Br) * pf)
#
#                         if A < B:
#                             self.fork_status["rational_changing_to_main"] += 1
#                             miner_change_chain[miner] = (chain2, chain1)
#                 if len(miner_change_chain) > 0:
#                     self.change_chains(miner_change_chain)
#



# ===================================================================================================================

    def get_miner_rewards(self, last_block, total_blocks):

        ignore_blocks = ignore_last_blocks
        current_block = last_block  # working backwards

        undercutter_counter = 0
        block_counter = 0
        miner_reward = {}
        while current_block.parent_block != None:
            if ignore_blocks > 0 :
                current_block = current_block.parent_block
                ignore_blocks -= 1
                continue
            if current_block.height < ignore_first_blocks:
                break
            block_counter += 1
            m = current_block.miner
            if isinstance(m, UndercutMiner):
                undercutter_counter += 1
            reward = current_block.fees
            try:
                miner_reward[m.name] += reward
            except:
                miner_reward[m.name] = reward

            current_block = current_block.parent_block

        print(miner_reward)
        miner_reward_ = {}
        for m in miner_reward:
            miner_reward_[m] = miner_reward[m]/sum(miner_reward.values())
        print(miner_reward_)

        return miner_reward_, block_counter, undercutter_counter, self.fork_status, total_blocks, self.mempool_stat, self.block_times, self.avoidance_status


# ===================================================================================================================
# ===================================================================================================================


##########################################################
##########################################################
##########################################################

# def setup(U_power, U2_power, H_power, cutoff,  iteration, avoidance, end_time=ending_timestamp):
#     print(U_power, U2_power, H_power, cutoff, iteration, avoidance)
#     iterations = range(30, 40, 1)
#     for iteration in iterations:
#         random.seed(a = iteration)
#         #time.sleep(random.random() * iteration * cutoff * H_power * U_power * 100)
#         #
#         fname = result_path+"undercutter_{0}_cuttoff_{1}_honest_{2}_iteration_{3}_avoidance_{4}".format(int(U_power*100), cutoff, int(H_power*100), iteration, avoidance)
#         if os.path.isfile(fname):
#             continue
#
#         # Create miners
#         m_u = UndercutMiner(name='undercutter', mining_power=U_power)
#         m_r = NormalMiner(name='rational', mining_power=1 - U_power - H_power)
#         if H_power == 1: # all honest
#             m_u = HonestMiner(name='undercutter_honest', mining_power=U_power)
#             m_h = HonestMiner(name='honest', mining_power=1 - U_power)
#             miners = [m_u, m_h]
#         elif H_power > 0:
#             m_h = HonestMiner(name='honest', mining_power=H_power)
#
#             miners = [m_u, m_h, m_r]
#         elif H_power == 0:
#             miners = [m_u, m_r]
#
#
#         # Create genesis block
#         genesis_block = Block(None, [], begining_timestamp-1, None, 0, 0, 0)
#
#         # Create simulation
#         s = Simulation(genesis_block=genesis_block, miners=miners, secondUndercutter=U2_power, start_timestamp=begining_timestamp, end_timestamp=end_time, cutoff=int(cutoff), undercutter_power=U_power, honest_power=H_power, avoidance=avoidance)
#         rewards, mainchain_blocks, undercutter_blocks, fork_status, total_blocks, mempool_stat, block_times = s.run()
#
#         # Print results
#         f_out = open(result_path+"undercutter_{0}_cuttoff_{1}_honest_{2}_iteration_{3}_avoidance_{4}".format(int(U_power*100), cutoff, int(H_power*100), iteration, avoidance), 'w')
#         f_out.write(json.dumps(rewards) + "\n")
#         f_out.write("num_total_blocks: " + str(total_blocks) + "\n")
#         f_out.write("num_total_blocks_mainchain: " + str(mainchain_blocks) + "\n")
#         f_out.write("num_blocks_undercutter_mainchain: " + str(undercutter_blocks) + "\n")
#         f_out.write("num_forks: " + str(fork_status["forks"]) + "\n")
#         f_out.write("num_failed_forks: " + str(fork_status["failed_forks"]) + "\n")
#         f_out.write("fork_conditions: " + json.dumps(fork_status["conditions"]) + "\n")
#         f_out.write("mempool_size: " + json.dumps(mempool_stat["size"]) + "\n")
#         f_out.write("mempool_fees: " + json.dumps(mempool_stat["fees"]) + "\n")
#         f_out.write("rational_miners_changing_to_fork: " + str(fork_status["rational_changing_to_fork"]) + "\n")
#         f_out.write("rational_miners_changing_to_main: " + str(fork_status["rational_changing_to_main"]) + "\n")
#         #print(fork_status["fork_len"])
#         #print(fork_status["main_len"])
#         #l = ''.join("("+str(e)+","+str(f)+ ")" for e,f in fork_status["fork_len"])
#         #f_out.write("forking block heights: " +l  + "\n")
#         f_out.write("average_failed_fork_len: " + str(statistics.mean(fork_status["fork_len"])) + "\n")
#         f_out.write("average_mainchain_fail_len: " + str(statistics.mean(fork_status["main_len"])) + "\n")
#         f_out.close()
#         gc.collect()



def setup_new(U_power, U2_power, H_power, cutoff,  iteration, avoidance, end_time=ending_timestamp):
    print(U_power, U2_power, H_power, cutoff, iteration, avoidance)
    iterations = range(0, 50, 1)
    for iteration in iterations:
        random.seed(a = iteration)
        #time.sleep(random.random() * iteration * cutoff * H_power * U_power * 100)
        #
        fname = result_path+"undercutter_{0}_cuttoff_{1}_honest_{2}_iteration_{3}_avoidance_{4}".format(int(U_power*100), cutoff, int(H_power*100), iteration, avoidance)
        if os.path.isfile(fname):
            continue

        # Create miners
        m_u = UndercutMiner(name='undercutter', mining_power=U_power)
        m_u2 = NormalMiner(name='undercutter2', mining_power=U2_power)


        if H_power == 1: # all honest
            m_u = HonestMiner(name='undercutter_honest', mining_power=U_power)
            m_h = HonestMiner(name='honest', mining_power=1 - U_power)
            miners = [m_u, m_h]

        else:
            R_power = 1 - U_power - U2_power - H_power
            m_r = []
            for i in range(0, int(R_power * 10)):
                m_r.append(NormalMiner(name='rational_{0}'.format(i), mining_power=R_power / int(R_power * 10)))

            if H_power == 0:
                miners = [m_u, m_u2] + m_r
            else:
                m_h = HonestMiner(name='honest', mining_power=H_power)
                miners = [m_u, m_u2, m_h] +  m_r

        print(miners)

        # Create genesis block
        genesis_block = Block(None, [], begining_timestamp-1, None, 0, 0, 0)

        # Create simulation
        s = Simulation(genesis_block=genesis_block, miners=miners, secondUndercutter=U2_power, start_timestamp=begining_timestamp, end_timestamp=end_time, cutoff=int(cutoff), undercutter_power=U_power, honest_power=H_power, avoidance=avoidance)
        rewards, mainchain_blocks, undercutter_blocks, fork_status, total_blocks, mempool_stat, block_times, avoidance_status = s.run()

        # Print results
        f_out = open(result_path+"undercutter_{0}_cuttoff_{1}_honest_{2}_iteration_{3}_avoidance_{4}".format(int(U_power*100), cutoff, int(H_power*100), iteration, avoidance), 'w')
        f_out.write(json.dumps(rewards) + "\n")
        f_out.write("num_total_blocks: " + str(total_blocks) + "\n")
        f_out.write("num_total_blocks_mainchain: " + str(mainchain_blocks) + "\n")
        f_out.write("num_blocks_undercutter_mainchain: " + str(undercutter_blocks) + "\n")
        f_out.write("num_forks: " + str(fork_status["forks"]) + "\n")
        f_out.write("num_failed_forks: " + str(fork_status["failed_forks"]) + "\n")
        f_out.write("fork_conditions: " + json.dumps(fork_status["conditions"]) + "\n")
        f_out.write("mempool_size: " + json.dumps(mempool_stat["size"]) + "\n")
        f_out.write("mempool_fees: " + json.dumps(mempool_stat["fees"]) + "\n")
        f_out.write("rational_miners_changing_to_fork: " + str(fork_status["rational_changing_to_fork"]) + "\n")
        f_out.write("rational_miners_changing_to_main: " + str(fork_status["rational_changing_to_main"]) + "\n")
        #print(fork_status["fork_len"])
        #print(fork_status["main_len"])
        #l = ''.join("("+str(e)+","+str(f)+ ")" for e,f in fork_status["fork_len"])
        #f_out.write("forking block heights: " +l  + "\n")
        f_out.write("average_failed_fork_len: " + str(statistics.mean(fork_status["fork_len"])) + "\n")
        f_out.write("average_mainchain_fail_len: " + str(statistics.mean(fork_status["main_len"])) + "\n")
        f_out.write("avoidance_conditions: " + json.dumps(avoidance_status) + "\n")
        f_out.close()
        gc.collect()




def parse_data():
    block_count = 0
    serial = 0
    txbydate = {}
    import glob
    data_folder = "/home/mohsen/get_new_data_undercutting/UPData/0506/"
    output_file = open("/home/mohsen/get_new_data_undercutting/tian_data_May15_to_June15", 'w')
    for f in glob.glob(data_folder + "*"):
        for l in open(f):
            st = json.loads(l)
            print(block_count, "\t", len(st['tx']))
            for tx in st['tx'][1:]:
                t = {'serial': serial, 'fee': tx['fee'], 'size': tx['size'], 'timestamp': tx['time'], 'block_id': tx['block_height']}
                try:
                    txbydate[tx['time']].append(t)
                except:
                    txbydate[tx['time']] = [t]
                serial += 1
            block_count += 1
            # if block_count > 20:
            #     break

    counter = 0
    for i in sorted(list(txbydate.keys())):
        for tx in txbydate[i]:
            output_file.write(json.dumps(tx)+"\n")
            print(counter)
            counter += 1
    print(serial)

#################################################################################################
#                           MAIN
#################################################################################################



if __name__== "__main__":
    # print("here")
    # parse_data()
    # exit()

    # total_size = 0
    # block_size = {}
    # tx_file = open("/home/mohsen/get_new_data_undercutting/tian_data_May15_to_June15")
    # min_time = math.inf
    # max_time = 0
    # for idx, l in enumerate(tx_file):
    #     print(idx)
    #     st=json.loads(l)
    #     try:
    #         block_size[st["block_id"]] += st["size"]
    #     except:
    #         block_size[st["block_id"]] = st["size"]
    #
    #     if st['timestamp'] < min_time:
    #         min_time = st['timestamp']
    #
    #     if st['timestamp'] > max_time:
    #         max_time = st['timestamp']
    #
    #     total_size += st['size']/1000000
    # print(block_size.values())
    # print("mean_size: ", statistics.mean(block_size.values()))
    # print("median_size: ", statistics.median(block_size.values()))
    # print(total_size)
    # print("start_time: ", min_time)
    # print("end_time: ", max_time)
    # exit()



    # end_time = load_txs(tx_file_path)
    # setup(0.45, 0, 2, 0, True, str(ending_timestamp))
    # exit()



    # end_time = load_txs(tx_file_path)
    # honest_percentage = [0, .1]
    # cutoff = [1, 2]
    # iterations = [1,2,3,4,5]
    # avoidance = [False, True]
    # for c in cutoff:
    #     for h in honest_percentage:
    #         for i in iterations:
    #             for a in avoidance:
    #                 block_times = setup(0.45, h, c, i, a, end_time)
    #                 print(sum(block_times)/len(block_times))
    #                 input("{0}_{1}_{2}_{3}".format(h, c, a, i))
    # exit()

    #input("starting_monero_june_last night2")


    # end_time = load_txs(tx_file_path)
    # setup_new(0.176, 0.153, 0.1, 1, 0, False, str(ending_timestamp))
    # exit()



    input("aqaq233")
    end_time = [load_txs(tx_file_path)]

    honest_percentage = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    honest_percentage = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    honest_percentage = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

    cutoff = [1,2]
    #cutoff = [2]

    avoidance = [True, False]
    #avoidance = [True]
    #avoidance = [False]

    undercutter=[0.176]
    undercutter=[0.499]
    #undercutter = [0.35]

    secondUndercutter = [.153]
    secondUndercutter = [.2]
    #secondUndercutter = [.29]

    iterations = range(1, 2, 1)

    paramlist_list = list(itertools.product(undercutter, secondUndercutter, honest_percentage, cutoff, iterations, avoidance, end_time))
    print(paramlist_list)
    print(len(paramlist_list))

    p = Pool(24)
    p.starmap(setup_new, paramlist_list)
    gc.collect()
    p.close()
    p.join()
    p.join()
    print("done")

