from block import Block
from mempool import Mempool
import copy
import time
import logging, sys
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
# algorithms: 1. greedy, 2. tricky  #change later

class Miner():
    def __init__(self, name, mining_power):
        self.name = name
        self.mining_power = mining_power
        self.profit = 0
        self.next_block = None
        self.avoidance_condition = 0
        self.avoidance_portion = 1



    def publish_next_block(self, timestamp, chains, currentChain):
        parent_block = currentChain.get_block(-1)
        t0=time.time()
        txs, txfees, block_size = self.get_nextBlock_txs(chains, currentChain, currentChain.get_block(-1))
        logging.debug("     get_txs: {0}".format(time.time() - t0))
        t0 = time.time()
        next_block = Block(miner=self, txs=txs, timestamp=timestamp, parent_block= parent_block, txfees=txfees, size=block_size, height=parent_block.height+1)
        logging.debug("     create_block: {0}".format(time.time() - t0))
        return next_block


    def add_txs_mempool(self, txs):
        self.mem_pool.add_txs(txs)

    def set_mempool(self, txs): # insert the txs into miners mempool
        # based on algorithm
        self.mem_pool.add_txs(txs)

    def get_mempool(self):  # return the miners mempool
        return self.mem_pool



class HonestMiner(Miner):
    def __init__(self, name, mining_power):
        super().__init__(name, mining_power)

    def get_nextBlock_txs(self, chains, forkChain, curr_block):
        # tx_blocks, block_fee, block_size = self.current_block.mempool.get_nextblock_txs()
        tx_blocks, block_fee, block_size = curr_block.get_nextblock_txs_avoidance(curr_block, self.avoidance_portion)
        return  tx_blocks, block_fee, block_size



class NormalMiner(Miner):
    def __init__(self, name, mining_power):
        super().__init__(name, mining_power)

    def get_nextBlock_txs(self, chains, forkChain, curr_block):
        #tx_blocks, block_fee, block_size = self.current_block.mempool.get_nextblock_txs()
        tx_blocks, block_fee, block_size = curr_block.get_nextblock_txs_avoidance(curr_block, self.avoidance_portion)
        return  tx_blocks, block_fee, block_size


class UndercutMiner(Miner):
    def __init__(self, name, mining_power):
        super().__init__(name, mining_power)
        self.fork_condition = 0
        self.infork = False

    def get_nextBlock_txs(self, chains, forkChain, curr_block):
        if self.infork and self.fork_condition > 0:
            for chain in chains:
                if chain == forkChain:
                    continue
                main_chain = chain
                break
            tx_blocks, block_fee, block_size = curr_block.get_nextblock_txs_fork(forkChain, self.fork_condition)
            self.fork_condition = 0
        else:
            #tx_blocks, block_fee, block_size = self.current_block.mempool.get_nextblock_txs()
            tx_blocks, block_fee, block_size = curr_block.get_nextblock_txs_avoidance(curr_block, self.avoidance_portion)

        return  tx_blocks, block_fee, block_size

