class Mempool():
    def __init__(self):
        self.txs = [] #Todo: make it dict with keys as serial nums
        self.fees = 0
        self.size = 0
        self.txs_map = {}


    def add_txs(self, txs):
        for tx in txs:
            self.txs.append(tx)
            self.fees += tx.fee
            self.size += tx.size
        self.txs = sorted(self.txs, key=lambda k: k.feerate, reverse=True)


    def remove_txs(self, new_block):
        for tx in new_block.txs:
            self.fees -= tx.fee
            self.size -= tx.size
            self.txs.remove(tx)


    def claimable_fees(self, num_blocks, block_size):
        fees = 0
        size = 0
        for tx in self.txs:
            if  size + tx.size > num_blocks * block_size:
                break
            fees += tx.fee
            size += tx.size
        return fees
