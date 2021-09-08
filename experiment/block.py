import time
import copy
from mempool import Mempool
import logging, sys

class Block():
    def __init__(self, miner, txs, timestamp, parent_block, txfees, size, height, type = "forked", mempool_index = -1):
        self.miner = miner
        self.parent_block = parent_block
        self.height = height
        self.timestamp = timestamp
        self.txs = txs
        self.fees = txfees
        self.size = size
        self.type = type
        self.mempool_index = mempool_index


    def get_nextblock_txs(self):
        fees = 0
        size = 0
        selected_txs = []
        for index, tx in enumerate(self.mempool.txs):
            if size + tx.size > self.mempool.block_size:  # check to not go over block size
                break
            fees += tx.fee
            size += tx.size
            selected_txs.append(tx)
        return selected_txs, fees, size


    def get_nextblock_txs_avoidance(self, new_block, portion):
        fees = 0
        size = 0
        selected_txs = []
        bandwidthset_fees = new_block.mempool.claimable_fees(1)
        for index, tx in enumerate(self.mempool.txs):
            if size + tx.size > self.mempool.block_size:  # check to not go over block size
                break
            if fees + tx.fee > portion * bandwidthset_fees:
                break
            fees += tx.fee
            size += tx.size
            selected_txs.append(tx)
        return selected_txs, fees, size


    def get_nextblock_txs_fork(self,fork_chain, fork_condition):
        selected_txs = []
        fees = 0
        size = 0

        if fork_condition == 1: # gamma negligible, then undercut with the first block being half of the main chain head
            for tx in fork_chain.main_chain_header_txs:
                if fees + tx.fee > 0.5 * fork_chain.main_chain_header_fees:
                    break
                selected_txs.append(tx)
                fees += tx.fee
                size += tx.size
        elif fork_condition == 2 or fork_condition == 3 or fork_condition == 6 or fork_condition == 7:  # undercut with the first block being the current bandwidth set, leaving everything in main chain head unclaimed
            for tx in fork_chain.main_chain_header_mempool.txs:
                if size + tx.size > self.mempool.block_size:  # check to not go over block size, this means it will claim the bandwidth set
                    break
                selected_txs.append(tx)
                fees += tx.fee
                size += tx.size
        elif fork_condition == 4: # gamma negligible, then undercut with the first block being 1/3 of the main chain head
            for tx in fork_chain.main_chain_header_txs:
                if fees + tx.fee > 0.33 * fork_chain.main_chain_header_fees:
                    break
                selected_txs.append(tx)
                fees += tx.fee
                size += tx.size
        elif fork_condition == 5: # undercut with the first block being 0.5 current bandwidth set
            for tx in fork_chain.main_chain_header_mempool.txs:
                if size + tx.size > self.mempool.block_size or fees + tx.fee > 0.5 * fork_chain.main_chain_header_bandwidthset_fees:
                    break
                selected_txs.append(tx)
                fees += tx.fee
                size += tx.size

        return selected_txs, fees, size

