from collections import deque
import copy
from miner import Miner, NormalMiner, UndercutMiner, HonestMiner
from mempool import Mempool
from block import Block
import time

class Chain():
    def __init__(self, miners, origin_block):
        self.prev_time = 0
        self.next_block_time = 0

        self.miners = miners
        self.mining_power_sum = 0
        self.mining_power_visible = 0
        self.miners_visible = []
        self.mining_weights = []

        self.mempool= Mempool()
        self.origin_block = origin_block

        self.next_block = origin_block

        self.avoidance_condition = 0
        self.avoidance_portion = 1

        self.blocks = [origin_block]

        # if main_chain is None:
        #     self.chain.append(origin_block)
        # else:
        #     for b in main_chain.chain:
        #         self.chain.append(b)
        #         if b == origin_block:
        #             break
        self.update_mining_power()

    def set_mempool(self,extending_chain):
        extending_chain_head_txs = extending_chain.get_block(-1).txs
        extending_chain_mempool_txs = extending_chain.mempool.txs
        self.mempool.add_txs(extending_chain_head_txs + extending_chain_mempool_txs)

    def set_next_block(self, blocksize):
        fees = 0
        size = 0
        selected_txs = []
        bandwidthset_fees = self.mempool.claimable_fees(num_blocks=1, block_size=blocksize)
        index = -1
        for index, tx in enumerate(self.mempool.txs):
            if size + tx.size > blocksize:  # check to not go over block size
                break
            if fees + tx.fee > self.avoidance_portion * bandwidthset_fees:
                break
            fees += tx.fee
            size += tx.size
            selected_txs.append(tx)
        print("++++++++++++++++++++++++++++++++++++BLOCK SIZE  ", size/1000000)
        block = Block(miner=None, txs=selected_txs, timestamp=None, parent_block=self.blocks[-1], txfees=fees, size=size, height= self.blocks[-1].height+1, type='general', mempool_index=index)
        self.next_block = block
        return block


    def publish_block(self, miner, timestamp):
        t0 = time.time()
        self.mempool.remove_txs(self.next_block)
        #print(self.next_block.type)
        #print("                 removing txs: ", time.time() - t0)
        self.next_block.miner = miner
        self.next_block.timestamp = timestamp
        self.blocks.append(self.next_block)
        try:
            self.blocks[-4].txs = []
        except:
            print("out of bounds************************************")
        return self.next_block

    def add_block(self, block):
        self.blocks.append(block)

    def len(self):
        return len(self.blocks)

    def get_block(self, index):
        return self.blocks[index]


    def update_mining_power(self):
        sum_weight = 0
        for miner in self.miners:
            sum_weight += miner.mining_power
        mining_weights = []
        for miner in self.miners:
            mining_weights.append(miner.mining_power/sum_weight)
        self.mining_weights = mining_weights
        self.mining_power_sum = sum_weight

    def get_mining_weights(self):
        self.update_mining_power()
        return self.mining_weights

    def get_mining_power_sum(self):
        self.update_mining_power()
        return self.mining_power_sum

    def get_rational_mining_power(self):
        self.update_mining_power()
        rational_power = 0
        for miner in self.miners:
            if isinstance(miner, NormalMiner):
                rational_power += miner.mining_power
        return rational_power

    def get_rational_mining_power_without_block(self):
        self.update_mining_power()
        rational_power = 0
        for miner in self.miners:
            if isinstance(miner, NormalMiner):
                rational_power += miner.mining_power
                for b in self.blocks:
                    if b.miner == miner:
                        rational_power -= miner.mining_power
                        break
        return rational_power
