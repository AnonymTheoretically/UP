from block import Block
from mempool import Mempool
import copy
import time


class Miner():
    def __init__(self, name, mining_power):
        self.name = name
        self.mining_power = mining_power
        self.profit = 0
        self.current_block = None
        self.mem_pool = Mempool()
        self.next_block = None

    def set_current_block(self, block):
        self.current_block = block


    def publish_next_block(self, timestamp):
        txs, txfees, block_size = self.get_nextBlock_txs()
        next_block = Block(miner=self, txs=txs, timestamp=timestamp, parent_block=self.current_block, txfees=txfees,
                           size=block_size, height=self.current_block.height+1)
        return next_block


    def add_txs_mempool(self, txs):
        self.mem_pool.add_txs(txs)

    def set_mempool(self, txs):
        self.mem_pool.add_txs(txs)

    def get_mempool(self):  # return the miners mempool
        return self.mem_pool



class HonestMiner(Miner):
    def __init__(self, name, mining_power):
        super().__init__(name, mining_power)
        self.threshold = 0

    def get_nextBlock_txs(self):
        tx_blocks, block_fee, block_size = self.current_block.mempool.get_nextblock_txs(threshold=self.threshold)
        return  tx_blocks, block_fee, block_size



class NormalMiner(Miner):
    def __init__(self, name, mining_power):
        super().__init__(name, mining_power)
        self.threshold = 0

    def get_nextBlock_txs(self):
        tx_blocks, block_fee, block_size = self.current_block.mempool.get_nextblock_txs(threshold=self.threshold)
        return  tx_blocks, block_fee, block_size


class UndercutMiner(Miner):
    def __init__(self, name, mining_power, mempool_threshold):
        super().__init__(name, mining_power)
        self.threshold = mempool_threshold
        self.infork = False
        self.first_block_fork = True

    def get_nextBlock_txs(self):
        if self.first_block_fork and self.infork:
            self.first_block_fork = False
            tx_blocks, block_fee, block_size = self.current_block.mempool.get_nextblock_txs(threshold=self.threshold)
        else:
            tx_blocks, block_fee, block_size = self.current_block.mempool.get_nextblock_txs(threshold=0)
        return  tx_blocks, block_fee, block_size

