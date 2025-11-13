
class Stock:
    def __init__(self, symbol: str, price: float):
        self.symbol = symbol
        self.price = price

    def __str__(self):
        return f"symbol: {self.symbol}, price: {self.price}"
