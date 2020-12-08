from block import Block
import logging, sys
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)


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
        txs, txfees, block_size = self.get_nextBlock_txs(chains, currentChain, currentChain.get_block(-1))
        next_block = Block(miner=self, txs=txs, timestamp=timestamp, parent_block= parent_block, txfees=txfees, size=block_size, height=parent_block.height+1)
        return next_block




class HonestMiner(Miner):
    def __init__(self, name, mining_power):
        super().__init__(name, mining_power)

    def get_nextBlock_txs(self, chains, forkChain, curr_block):
        tx_blocks, block_fee, block_size = curr_block.get_nextblock_txs_avoidance(curr_block, self.avoidance_portion)
        return  tx_blocks, block_fee, block_size



class NormalMiner(Miner):
    def __init__(self, name, mining_power):
        super().__init__(name, mining_power)

    def get_nextBlock_txs(self, chains, forkChain, curr_block):
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
            tx_blocks, block_fee, block_size = curr_block.get_nextblock_txs_avoidance(curr_block, self.avoidance_portion)

        return  tx_blocks, block_fee, block_size

