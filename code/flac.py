import crcmod
import struct

from utility import *

BLOCK_SIZE = 4096       # Samples per block
SAMPLE_RATE = 44100     # Hz
SAMPLE_SIZE = 16        # Bits per sample
NUM_CHANNEL = 2     

crc8 = crcmod.predefined.mkPredefinedCrcFun('crc-8')
crc16 = crcmod.predefined.mkPredefinedCrcFun('crc-16-buypass')

class Stream:
    def __init__(self, metadata_blocks, frames):
        self.metadata_blocks = metadata_blocks
        self.frames = frames
    
    def get_bytes(self):
        return b'fLaC' + \
               b''.join([block.get_bytes() for block in self.metadata_blocks]) + \
               b''.join([frame.get_bytes() for frame in self.frames])

class MetadataBlock:
    def __init__(self, metadata_block_header, metadata_block_data):
        self.metadata_block_header = metadata_block_header
        self.metadata_block_data = metadata_block_data

    def get_bytes(self):
        return self.metadata_block_header.get_bytes() + \
               self.metadata_block_data.get_bytes()

class MetadataBlockHeader:
    def __init__(self, last_metadata_block, block_type, length):
        self.last_metadata_block = last_metadata_block
        self.block_type = block_type
        self.length = length

    def get_bytes(self):
        bits = bitarray(32)

        bits[0] = self.last_metadata_block
        bits[1:8] = bitarray_from_int(self.block_type, 7)
        bits[8:32] = bitarray_from_int(self.length, 24)

        return bits.tobytes()

class MetadataBlockStreamInfo:
    def __init__(self, num_samples, md5_digest):
        self.num_samples = num_samples
        self.md5_digest = md5_digest

    def get_bytes(self):
        bits = bitarray(144)

        bits[0:16] = bitarray_from_int(BLOCK_SIZE, 16)
        bits[16:32] = bitarray_from_int(BLOCK_SIZE, 16)
        bits[32:56] = 0
        bits[56:80] = 0
        bits[80:100] = bitarray_from_int(SAMPLE_RATE, 20)
        bits[100:103] = bitarray_from_int(NUM_CHANNEL-1, 3)
        bits[103:108] = bitarray_from_int(SAMPLE_SIZE-1, 5)
        bits[108:144] = bitarray_from_int(self.num_samples, 36)

        return bits.tobytes() + self.md5_digest

class Frame:
    def __init__(self, frame_number, num_samples, subframes):
        self.frame_number = frame_number
        self.num_samples = num_samples
        self.subframes = subframes

    def get_header_bytes(self):
        bits = bitarray(32)                     

        bits[0:14] = bitarray('11111111111110')         # Sync code
        bits[14] = 0                                    # Reserved
        bits[15] = 0                                    # Blocking strategy (fixed-blocksize)
        bits[16:20] = bitarray('1100')                  # 256 * (2^n-8) samples: 12 --> 4096 blocksize
        bits[20:24] = bitarray('1001')                  # Sample rate 44100 Hz
        bits[24:28] = bitarray('0001')                  # Channel assignment (stereo) / numchannel - 1
        bits[28:31] = bitarray('100')                   # Sample size (16 bits per sample)
        bits[31] = 0                                    # Mandatory value

        frame_number_bits = utf8_encoded_bitarray_from_int(self.frame_number)

        custom_block_size_bits = bitarray()

        # L'ultimo blocco potrebbe essere più piccolo di BLOCKSIZE
        if self.num_samples != BLOCK_SIZE:
            bits[16:20] = bitarray('0111')              # get 16 bit (blocksize-1) from end of header
            custom_block_size_bits = bitarray_from_int(self.num_samples - 1, 16)

        crc_input = (bits + frame_number_bits + custom_block_size_bits).tobytes()
        crc_bytes = bytes((crc8(crc_input),))

        return crc_input + crc_bytes
    
    def get_subframe_and_padding_bytes(self):
        subframe_bits = sum([subframe.get_bits() for subframe in self.subframes], bitarray())

        num_padding_bits = 0

        if subframe_bits.length() % 8:
            num_padding_bits = 8 - (subframe_bits.length() % 8)
        
        padding_bits = bitarray(num_padding_bits)
        padding_bits.setall(0)

        return (subframe_bits + padding_bits).tobytes()
    
    def get_footer_bytes(self):
        crc_input = self.get_header_bytes() + self.get_subframe_and_padding_bytes()
        crc_bytes = struct.pack('>H', crc16(crc_input))

        return crc_bytes

    def get_bytes(self):
        return self.get_header_bytes() + \
               self.get_subframe_and_padding_bytes() + \
               self.get_footer_bytes()

class Subframe:
    def __init__(self):
        # Subframe è composta da header e data
        self.header_bits = bitarray(8)
        self.data_bits = bitarray()

        self.header_bits[0] = 0         # Mandatory value
        self.header_bits[1:7] = 0       # Riempiti dopo in base al tipo di subframe
        self.header_bits[7] = 0         # Wasted bits

    def __len__(self):
        return self.get_bits().length()

    def get_bits(self):
        return self.header_bits + self.data_bits

class SubframeConstant(Subframe):
    def __init__(self, constant):
        super().__init__()

        self.header_bits[1:7] = bitarray('000000')  # Constant subframe
        self.data_bits = bitarray_from_signed(constant, SAMPLE_SIZE)

class SubframeVerbatim(Subframe):
    def __init__(self, samples):
        super().__init__()

        self.header_bits[1:7] = bitarray('000001')      # Verbatim subframe
        
        verbatim_sample_bytes = struct.pack('>' + str(len(samples)) + 'h', *samples)
        self.data_bits.frombytes(verbatim_sample_bytes)

class SubframeFixed(Subframe):
    def __init__(self, predictor_order, warmup_samples, residual):
        super().__init__()

        self.header_bits[1:4] = bitarray('001')         # Fixed subframe
        self.header_bits[4:7] = bitarray_from_int(predictor_order, 3)

        warmup_sample_bits = bitarray()

        for sample in warmup_samples:
            warmup_sample_bits.extend(bitarray_from_signed(sample, SAMPLE_SIZE))

        self.data_bits = warmup_sample_bits + residual.get_bits()

class Residual:
    def __init__(self, coding_method, partitioned_rice):
        self.coding_method = coding_method
        self.partitioned_rice = partitioned_rice
    
    def get_bits(self):
        coding_method_bits = bitarray('00') if self.coding_method == 0 else bitarray('01')

        return coding_method_bits + self.partitioned_rice.get_bits()

class PartitionedRice:
    def __init__(self, partition_order, rice_partition):
        self.partition_order = partition_order
        self.rice_partition = rice_partition

    def get_bits(self):
        # Partition order 4 bits
        partition_order_bits = bitarray_from_int(self.partition_order, 4)

        return sum([partition.get_bits() for partition in self.rice_partition], partition_order_bits)

class Rice2Partition:
    def __init__(self, parameter, residual_signal):
        self.parameter = parameter
        self.residual_signal = residual_signal

    def get_bits(self):
        # Prendiamo in considerazione solo parametri minori di 31
        assert self.parameter < 31  
        # Quindi calcoliamo l'array di bit per rappresentare il parametro (necessitano 5 bit)
        parameter_bits = bitarray_from_int(self.parameter, 5)
        
        # Lista dei campioni codificati
        encoded_samples = list()

        # Per ogni campione del residual signal (array delle differenze tra predizioni e valori reali)
        for sample in self.residual_signal:

            mapped_sample = -2 * sample - 1 if sample < 0 else 2 * sample

            # Split dei bits del mapped sample in due metà:
            # high-order-bits e low-order-bits
            mask = (1 << self.parameter) - 1
            # And logico tra il mapped e la mask
            low_order_bits = mapped_sample & mask
            # Shift a destra del mapped di 'parameter' bits
            high_order_bits = mapped_sample >> self.parameter

            # Allocazione del numero necessario di bits di low_order
            low_order_bitarray = bitarray_from_int(low_order_bits, self.parameter)
            # Allocazione del numero necessario di bits di high_order
            high_order_bitarray = bitarray(high_order_bits)
            # Li setto tutti a 0
            high_order_bitarray.setall(0)

            encoded_sample = high_order_bitarray + bitarray('1') + low_order_bitarray
            encoded_samples.append(encoded_sample)

        return sum(encoded_samples, parameter_bits)