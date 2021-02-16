from bitarray import bitarray

def bitarray_from_int(i, width):
    assert i < 2**width

    if width == 0:
        return bitarray()

    return bitarray(('{:0' + str(width) + 'b}').format(i))

def bitarray_from_signed(i, width):
    assert i < 2**(width-1)
    assert i >= -2**(width-1)

    if width == 0:
        assert i == 0
        return bitarray()
    
    i -= (i >> (width-1)) << width
    return bitarray(('{:0' + str(width) + 'b}').format(i))

def utf8_encoded_bitarray_from_int(i):
    # i < 2**7
    if i < 0x80:
        return bitarray_from_int(i, 8)

    # i < 2**11
    if i < 0x800:
        bits = bitarray(16)

        bits[0:8]   = bitarray_from_int(0xC0 | (i >> 6), 8)
        bits[8:16]  = bitarray_from_int(0x80 | (i & 0x3F), 8)

        return bits

    # i < 2**16
    if i < 0x10000:
        bits = bitarray(24)

        bits[0:8]   = bitarray_from_int(0xE0 | (i >> 12), 8)
        bits[8:16]  = bitarray_from_int(0x80 | ((i >> 6) & 0x3F), 8)
        bits[16:24] = bitarray_from_int(0x80 | (i & 0x3F), 8)

        return bits

    # i < 2**21
    if i < 0x200000:
        bits = bitarray(32)

        bits[0:8]   = bitarray_from_int(0xF0 | ((i >> 18)), 8)
        bits[8:16]  = bitarray_from_int(0x80 | ((i >> 12) & 0x3F), 8)
        bits[16:24] = bitarray_from_int(0x80 | ((i >> 6) & 0x3F), 8)
        bits[24:32] = bitarray_from_int(0x80 | (i & 0x3F), 8)

        return bits

    # i < 2**26
    if i < 0x4000000:
        bits = bitarray(40)

        bits[0:8]   = bitarray_from_int(0xF0 | ((i >> 24)), 8)
        bits[8:16]  = bitarray_from_int(0x80 | ((i >> 18) & 0x3F), 8)
        bits[16:24] = bitarray_from_int(0x80 | ((i >> 12) & 0x3F), 8)
        bits[24:32] = bitarray_from_int(0x80 | ((i >> 6) & 0x3F), 8)
        bits[32:40] = bitarray_from_int(0x80 | (i & 0x3F), 8)

        return bits

    # i < 2**31
    if i < 0x80000000:
        bits = bitarray(40)

        bits[0:8]   = bitarray_from_int(0xF0 | ((i >> 24)), 8)
        bits[8:16]  = bitarray_from_int(0x80 | ((i >> 18) & 0x3F), 8)
        bits[16:24] = bitarray_from_int(0x80 | ((i >> 12) & 0x3F), 8)
        bits[24:32] = bitarray_from_int(0x80 | ((i >> 6) & 0x3F), 8)
        bits[32:40] = bitarray_from_int(0x80 | (i & 0x3F), 8)

        return bits

    assert False, "We shouldn't need to encode any integers that require more than 31 bits"