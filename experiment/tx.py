class Tx():
    def __init__(self, timestamp, size, fee, serial):
        self.timestamp = timestamp
        self.size = size
        self.fee = fee
        self.feerate = fee/size
        self.serial = serial
