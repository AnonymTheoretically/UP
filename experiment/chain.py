from collections import deque

class Chain():
    def __init__(self, miners, main_chain, origin_block):
        self.origin_block = origin_block
        self.miners = miners
        self.mining_power_sum = 0
        self.mining_power_visible = 0
        self.miners_visible = []
        self.mining_weights = []
        self.chain = deque()
        if main_chain is None:
            self.chain.append(origin_block)
        else:
            for b in main_chain.chain:
                self.chain.append(b)
                if b == origin_block:
                    break
        self.update_mining_power()


    def add_block(self, block):
        self.chain.append(block)
        if len(self.chain) > 7:
            self.remove_block()

    def remove_block(self):
        self.chain.popleft()

    def len(self):
        return len(self.chain)

    def get_block(self, index):
        return self.chain[index]

    def update_visible_miners(self, start_index):
        self.miners_visible = []
        miners = set()
        for block in list(self.chain)[start_index + 1: ]:
            miners.add(block.miner)

        self.mining_power_visible = 0
        for miner in miners:
            self.mining_power_visible += miner.mining_power
            self.miners_visible.append(miner)


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
        self.update_mining_powers()
        return self.mining_weights

    def get_mining_power_sum(self):
        self.update_mining_powers()
        return self.mining_power_sum

    def get_visible_mining_power(self, start_index):
        self.update_visible_miners(start_index)
        return self.mining_power_visible

    def update_mining_powers(self):
        self.update_mining_power()


    def get_profits(self, start, end):
        profit_sum = 0
        for block in list(self.chain)[start+1:end+1]:
            profit_sum += block.txfees
        return profit_sum