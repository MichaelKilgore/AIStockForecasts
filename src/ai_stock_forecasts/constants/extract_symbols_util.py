

def get_symbols():
    with open("symbols.txt", "r") as f:
        symbols = [line.strip() for line in f]

    return symbols

if __name__  == "__main__":
    x = get_symbols()

    print(set(x))
    print(len(set(x)))
    print(len(x))