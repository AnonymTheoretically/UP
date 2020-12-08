class Tx():
    def __init__(self, timestamp, size, fee, serial):
        self.timestamp = timestamp
        self.size = size
        self.fee = fee
        self.feerate = fee/size
        self.serial = serial

    def __eq__(self, other):
        return self.serial == other.serial

    def __lt__(self, other):
        return (self.serial < other.serial)

    def __gt__(self, other):
        return (self.serial > other.serial)