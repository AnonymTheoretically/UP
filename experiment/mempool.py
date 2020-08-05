class Mempool():
    def __init__(self):
        self.txs = [] #Todo: change to a max heap for better performance
        self.block_size = 2000000


    def add_tx(self, new_tx):
        for index, tx in enumerate(self.txs):
            if new_tx.feerate > tx.feerate:
                self.txs.insert(index, new_tx)
                return
        self.txs.append(new_tx)

    def add_txs(self, txs): # batch addition
        # for tx in txs:
        #     self.add_tx(tx)
        for tx in txs:
            self.txs.append(tx)
        self.txs = sorted(self.txs, key=lambda k:k.feerate)



    def remove_tx(self, removing_tx):
        for tx in self.txs:
            if tx.serial == removing_tx.serial:
                self.txs.remove(tx)
                return

    def remove_txs(self, txs): # batch deletion
        for tx in txs:
            self.remove_tx(tx)

    def get_nextblock_txs(self, threshold):
        total_max_fee = 0
        total_size = 0
        for tx in self.txs:
            if total_size + tx.size > self.block_size:  # check to not go over block size
                break
            total_max_fee += tx.fee
            total_size += tx.size


        leave_out_fee = threshold * total_max_fee
        total_fee = 0
        total_size = 0
        start_index = 0
        current_index = 0
        while leave_out_fee > 0:
            leave_out_fee -= self.txs[start_index].fee
            start_index += 1

        for tx in self.txs[start_index : ]:
            if total_size + tx.size > self.block_size:  # check to not go over block size
                break
            total_fee += tx.fee
            total_size += tx.size
            current_index += 1

        selected_txs = self.txs[start_index:current_index+1]
        #mempool_witout_txs = self.txs[0:start_index] + self.txs[current_index:]
        #self.txs = self.txs[0:start_index] + self.txs[current_index:] # remove the selected transactions from mempool
        return selected_txs, total_fee, total_size


