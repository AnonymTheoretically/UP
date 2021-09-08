class Mempool():
    def __init__(self):
        self.txs = [] #Todo: make it dict with keys as serial nums
        #self.block_size = 1183248
        #self.block_size = 1266149
        #self.block_size = 2000000
        #self.block_size = 4000000
        #self.block_size = 100
        self.fees = 0
        self.size = 0
        self.txs_map = {}


    def add_txs(self, txs):
        for tx in txs:
            self.txs.append(tx)
            self.fees += tx.fee
            self.size += tx.size
        self.txs = sorted(self.txs, key=lambda k: k.feerate, reverse=True)

    # def add_txs(self, txs):
    #     for tx in txs:
    #         self.txs_map[tx.serial] = tx
    #         self.fees += tx.fee
    #         self.size += tx.size
    #     self.txs = sorted(list(self.txs_map.values()), key=lambda k:k.feerate, reverse=True)
    #


    # def remove_txs(self, new_block):
    #     old_len = len(self.txs)
    #     if new_block.type == "general":
    #         print("here")
    #         if len(new_block.txs) == len(self.txs):
    #             self.txs = []
    #             self.fees = 0
    #             self.size = 0
    #             print(len(self.txs))
    #             print(old_len)
    #             print(len(new_block.txs))
    #             print(new_block.mempool_index)
    #             print("VA CHECKING")
    #
    #         elif new_block.mempool_index == -1 and len(new_block.txs) == 0:
    #             print("empty Block")
    #             return
    #         else:
    #             self.txs = self.txs[new_block.mempool_index : ]
    #             self.fees -= new_block.fees
    #             self.size -= new_block.size
    #             print(len(self.txs))
    #             print(old_len)
    #             print(len(new_block.txs))
    #             print(new_block.mempool_index)
    #             print("VA CHECKING2")
    #     else:
    #         for tx in new_block.txs:
    #             self.fees -= tx.fee
    #             self.size -= tx.size
    #             self.txs.remove(tx)
    #         print("here2")
    #     if len(self.txs) != old_len - len(new_block.txs):
    #         print(len(self.txs))
    #         print(old_len)
    #         print(len(new_block.txs))
    #         input("VA MOSIBATA")
    #     #l = [x for x in self.txs if x not in txs]
    #     #self.txs = l


    #
    # def remove_txs(self, new_block):
    #     old_len = len(self.txs)
    #     for index, tx in enumerate(new_block.txs):
    #         del self.txs_map[tx.serial]
    #         self.fees -= tx.fee
    #         self.size -= tx.size
    #
    #     self.txs = sorted(self.txs_map.values(), key=lambda k:k.feerate, reverse=True)
    #     if len(self.txs) != old_len - len(new_block.txs):
    #         input("removing txs from mempool needs fixing")


    def remove_txs(self, new_block):
        old_len = len(self.txs)
        for tx in new_block.txs:
            self.fees -= tx.fee
            self.size -= tx.size
            self.txs.remove(tx)
            # l = [x for x in self.txs if x not in txs]
            # self.txs = l
        if len(self.txs) != old_len - len(new_block.txs):
            input("removing txs from mempool needs fixing")

    def claimable_fees(self, num_blocks, block_size):
        fees = 0
        size = 0
        for tx in self.txs:
            if  size + tx.size > num_blocks * block_size:
                break
            fees += tx.fee
            size += tx.size
        return fees
