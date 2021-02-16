import struct
import sys
from bitinputstream import BitInputStream

FIXED_PREDICTION_COEFFICIENTS = (
	(),
	(1,),
	(2, -1),
	(3, -3, 1),
	(4, -6, 4, -1),
)

def main(argv):
    # Apro il file Flac utilizzando una classe di supporto chiamata BitInputStream (esterna)
    # E lo decodifico
    with BitInputStream(open(argv[1], "rb")) as inp:
        with open(argv[2], "wb") as out:
            decode(inp, out)

def decode(inp, out):
    # Leggo il file flac e verifico che i primi 32 bit siano coerenti con il formato flac
    if inp.read_uint(32) != 0x664C6143:
        raise ValueError("Invalid fLaC file!")
    
    samplerate = None
    last = False
    # Ciclo fin quando non trovo il bit "last"
    # Leggo il flusso di input bit per bit
    # Salvo tutto ciò che mi interessa per la conversione (samplerate, numchannels, ...)
    while not last:
        last = inp.read_uint(1) != 0
        typo = inp.read_uint(7)
        length = inp.read_uint(24)
        if typo == 0:
            inp.read_uint(16)
            inp.read_uint(16)
            inp.read_uint(24)
            inp.read_uint(24)
            samplerate = inp.read_uint(20)
            numchannels = inp.read_uint(3) + 1
            samplesize = inp.read_uint(5) + 1
            numsamples = inp.read_uint(36)
            inp.read_uint(128)
        else:
            for i in range(length):
                inp.read_uint(8)
    
    # Verifico che il samplerate non sia nullo
    if samplerate is None:
        raise ValueError("Stream info metadata block absent!")
    
    # Verifico che i bit_per_sample siano multipli di 8
    if samplesize % 8 != 0:
        raise RuntimeError("Sample size not supported!")
    
    # Creo il flusso da decodificare 
    stream = WaveStream(samplesize, samplerate, numchannels, numsamples)

    # Chiamo la funzione write che decodifica il flusso e scrive il nuovo file
    write_stream(inp, stream, out)

def write_stream(inp, stream, out):
    # Scrivo l'header del file WAV
    out.write(b"RIFF")
    sampledatalen = stream.num_samples * stream.num_channels * (stream.sample_size // 8)
    out.write(struct.pack("<I", sampledatalen + 36))
    out.write(b"WAVE")
    out.write(b"fmt ")
    out.write(struct.pack("<IHHIIHH", 16, 0x0001, stream.num_channels, stream.sample_rate,
                stream.sample_rate * stream.num_channels * (stream.sample_size // 8),
                stream.num_channels * (stream.sample_size // 8), stream.sample_size))
    out.write(b"data")
    out.write(struct.pack("<I", sampledatalen))

    # Decodifico tutti i frames
    while decode_frame(inp, stream.num_channels, stream.sample_size, out):
        pass

def decode_frame(inp, num_channels, sample_size, out):
    temp = inp.read_byte()
    if temp == -1:
        return False
    sync = temp << 6 | inp.read_uint(6)
    if sync != 0x3FFE:
        raise ValueError("Sync code expected!")

    inp.read_uint(1)
    inp.read_uint(1)
    blocksizecode = inp.read_uint(4)
    sampleratecode = inp.read_uint(4)
    chanasgn = inp.read_uint(4)
    inp.read_uint(3)
    inp.read_uint(1)

    temp = inp.read_uint(8)
    while temp >= 0b11000000:
        inp.read_uint(8)
        temp = (temp << 1) & 0xFF
    
    if blocksizecode == 1:
        blocksize = 192
    elif 2 <= blocksizecode <= 5:
        blocksize = 576 << blocksizecode - 2
    elif blocksizecode == 6:
        blocksize = inp.read_uint(8) + 1
    elif blocksizecode == 7:
        blocksize = inp.read_uint(16) + 1
    elif 8 <= blocksizecode <= 15:
        blocksize = 256 << (blocksizecode - 8)

    if sampleratecode == 12:
        inp.read_uint(8)
    elif sampleratecode in (13, 14):
        inp.read_uint(16)

    inp.read_uint(8)

    # Decode each channel's subframe, then skip footer
    samples = decode_subframes(inp, blocksize, sample_size, chanasgn)
    inp.align_to_byte()
    inp.read_uint(16)

    # Write the decoded samples
    numbytes = sample_size // 8
    addend = 128 if sample_size == 8 else 0
    for i in range(blocksize):
        for j in range(num_channels):
            out.write(struct.pack("<i", samples[j][i] + addend)[ : numbytes])
    return True

def decode_subframes(inp, blocksize, sample_size, chanasgn):
    if 0 <= chanasgn <= 7:
        return [decode_subframe(inp, blocksize, sample_size) for _ in range(chanasgn + 1)]
    elif 8 <= chanasgn <= 10:
        temp0 = decode_subframe(inp, blocksize, sample_size + (1 if (chanasgn == 9) else 0))
        temp1 = decode_subframe(inp, blocksize, sample_size + (0 if (chanasgn == 9) else 1))
        if chanasgn == 8:
            for i in range(blocksize):
                temp1[i] = temp0[i] - temp1[i]
        elif chanasgn == 9:
            for i in range(blocksize):
                temp0[i] += temp1[i]
        elif chanasgn == 10:
            for i in range(blocksize):
                side = temp1[i]
                right = temp0[i] - (side >> 1)
                temp1[i] = right
                temp0[i] = right + side
        return [temp0, temp1]
    else:
        raise ValueError("Reserved channel assignment")

def decode_subframe(inp, blocksize, sample_size):
    inp.read_uint(1)
    type = inp.read_uint(6)
    shift = inp.read_uint(1)
    if shift == 1:
        while inp.read_uint(1) == 0:
            shift += 1
    sample_size -= shift

    if type == 0:  # Constant coding
        result = [inp.read_signed_int(sample_size)] * blocksize
    elif type == 1:  # Verbatim coding
        result = [inp.read_signed_int(sample_size) for _ in range(blocksize)]
    elif 8 <= type <= 12:
        result = decode_fixed_prediction_subframe(inp, type - 8, blocksize, sample_size)
    elif 32 <= type <= 63:
        result = decode_linear_predictive_coding_subframe(inp, type - 31, blocksize, sample_size)
    else:
        raise ValueError("Reserved subframe type")
    return [(v << shift) for v in result]

def decode_fixed_prediction_subframe(inp, predorder, blocksize, sample_size):
    result = [inp.read_signed_int(sample_size) for _ in range(predorder)]
    decode_residuals(inp, blocksize, result)
    restore_linear_prediction(result, FIXED_PREDICTION_COEFFICIENTS[predorder], 0)
    return result

def decode_residuals(inp, blocksize, result):
    method = inp.read_uint(2)
    if method >= 2:
        raise ValueError("Reserved residual coding method")
    parambits = [4, 5][method]
    escapeparam = [0xF, 0x1F][method]

    partitionorder = inp.read_uint(4)
    numpartitions = 1 << partitionorder
    if blocksize % numpartitions != 0:
        raise ValueError("Block size not divisible by number of Rice partitions")

    for i in range(numpartitions):
        count = blocksize >> partitionorder
        if i == 0:
            count -= len(result)
        param = inp.read_uint(parambits)
        if param < escapeparam:
            result.extend(inp.read_rice_signed_int(param) for _ in range(count))
        else:
            numbits = inp.read_uint(5)
            result.extend(inp.read_signed_int(numbits) for _ in range(count))

def restore_linear_prediction(result, coefs, shift):
    for i in range(len(coefs), len(result)):
        result[i] += sum((result[i - 1 - j] * c) for (j, c) in enumerate(coefs)) >> shift

class WaveStream:
    def __init__(self, sample_size, sample_rate, num_channels, num_samples):
        self.sample_size = sample_size
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.num_samples = num_samples

if __name__ == "__main__":
    main(sys.argv)