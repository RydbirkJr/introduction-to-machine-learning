

def print_bits(length, set, string=''):
    for item in set:
        temp = string + item
        if length == 1:
            print temp
        else:
            print_bits(length-1, set, temp)

print_bits(12, ['1', '0'])

