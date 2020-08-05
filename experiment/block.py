import copy
from mempool import Mempool

class Block():
    def __init__(self, miner, txs, timestamp, parent_block, txfees, size, height):
        self.miner = miner
        self.timestamp = timestamp
        self.parent_block = parent_block
        self.next_block = None
        self.txfees = txfees
        self.size = size
        self.confirmed = 0
        self.height = height
        self.txs = txs
        self.forked = False

        try:
            self.mempool = copy.deepcopy(parent_block.mempool)
        except:
            self.mempool = Mempool()

        try:
            self.parent_mempool = copy.deepcopy(parent_block.mempool)
        except:
            self.parent_mempool = Mempool()

    def set_block_confirmed(self):
        self.confirmed = 1

    def get_miner(self):
        return self.miner

    def add_txs_mempool(self, txs):
        self.mempool.add_txs(txs)

    def remove_txs_mempool(self, txs):
        self.mempool.remove_txs(txs)