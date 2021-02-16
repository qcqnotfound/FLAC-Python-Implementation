import sys
import wave
import hashlib
import math
import array
import numpy as np 
from flac import *

BLOCK_SIZE = 4096       # Samples per block
SAMPLE_RATE = 44100     # Hz
SAMPLE_SIZE = 16        # Bits per sample
NUM_CHANNEL = 2   

MAX_FIXED_PREDICTOR_ORDER = 4

# Tipologie di blocco Flac (uso solo streaminfo)
BLOCK_TYPE_STREAMINFO = 0
#BLOCK_TYPE_PADDING = 1
#BLOCK_TYPE_APPLICATION = 2
#BLOCK_TYPE_SEEKTABLE = 3
#BLOCK_TYPE_VORBIS_COMMENT = 4
#BLOCK_TYPE_CUESHEET = 5
#BLOCK_TYPE_PICTURE = 6

# Tipologie di partition rice method (uso solo la 2)
#RESIDUAL_CODING_METHOD_PARTITION_RICE = 0
RESIDUAL_CODING_METHOD_PARTITION_RICE2 = 1

def main(argv):
    # Leggo il file di input e creo il flusso da codificare
    wave_stream = read_wave(argv[1])
    # Codifico il flusso di input e restituisco il nuovo flusso
    stream = encode_wave_stream(wave_stream)
    # Scrivo il file flac
    write_stream(stream, argv[2])

def read_wave(input_path):
    # Apro il file wav
    input_file = wave.open(input_path, 'rb')

    # Bits per sample
    sample_size = input_file.getsampwidth() * 8
    # Frame rate
    sample_rate = input_file.getframerate()
    # Numero di canali
    num_channels = input_file.getnchannels()
    # Numero di frames
    num_samples = input_file.getnframes()
    # Numero frames totali
    num_interleaved_samples = num_channels * num_samples

    # Restrizioni al tipo di file di input
    assert sample_size == SAMPLE_SIZE, "Only 16 bit"
    assert sample_rate == SAMPLE_RATE, "Only 44.1 Hz"
    assert num_channels == NUM_CHANNEL, "Only stereo input"

    # Frames (stringa)
    raw_frames = input_file.readframes(num_samples)

    # Chiudo il file
    input_file.close()

    # MD5
    md5_digest = hashlib.md5(raw_frames).digest()

    # "Estraggo" i valori numerici dei campioni
    # Gli indici pari contengono i valori del primo canale
    # Gli indici dispari quelli del secondo
    # '<' --> little-endian, 'h' --> short int
    interleaved_samples = struct.unpack('<' + str(num_interleaved_samples) + 'h', raw_frames)

    # Creo una lista contenente tante liste quanti sono i canali    
    channels = [list() for i in range(num_channels)]

    # Metto ogni campione nella lista giusta (pari -> primo canale, dispari -> secondo canale)
    for index, sample in enumerate(interleaved_samples):
        channels[index % num_channels].append(sample)

    # Creo il flusso da codificare utilizzando la classe WaveStream
    wave_stream = WaveStream(sample_rate, (sample_size/8), [array.array('h', channel) for channel in channels], md5_digest)
    
    return wave_stream

def encode_wave_stream(wave_stream):
    # Inizializzo la lista dei frames
    frames = list()

    # Creo il mio nuovo flusso
    for sample_index in range(0, wave_stream.num_samples, BLOCK_SIZE):
        # Ogni frame ha dimensione "BLOCK_SIZE" quindi ogni frame avrà questo "indice":
        frame_number = sample_index // BLOCK_SIZE
        # Inizializzo lista dei subframes
        subframes = list()

        # Flac ha quattro tipi di subframes (ne implemento 3):
        for channel in wave_stream.channels:
            subframe_candidates = list()

            # Constant
            subframe_candidates.append(make_subframe_constant(channel, sample_index))
            # Verbatim
            subframe_candidates.append(make_subframe_verbatim(channel, sample_index))
            # Fixed
            for fixed_predictor_order in range(MAX_FIXED_PREDICTOR_ORDER + 1):
                subframe_candidates.append(make_subframe_fixed(channel, sample_index, fixed_predictor_order))

            subframe_candidates = filter(None, subframe_candidates)
            smallest_subframe = min(subframe_candidates, key=len)
            # Conserverò solamente quello più piccolo
            subframes.append(smallest_subframe)

        # Calcolo il numero di campioni nel frame
        num_samples_in_frame = (wave_stream.num_samples - sample_index) if (wave_stream.num_samples - sample_index) < BLOCK_SIZE else BLOCK_SIZE

        # Creo il nuovo frame
        frame = Frame(frame_number, num_samples_in_frame, subframes)

        # Lo aggiungo alla lista dei frame
        frames.append(frame)

    # Aggiungo il blocco di metadati riguardanti le info sul flusso
    metadata_block_stream_info = MetadataBlockStreamInfo(wave_stream.num_samples, wave_stream.md5_digest)
    # Aggiungo l'header
    metadata_block_header = MetadataBlockHeader(True, BLOCK_TYPE_STREAMINFO, len(metadata_block_stream_info.get_bytes()))
    # Costruisco il blocco "Metadati" in generale composto da header + stream info
    metadata_block = MetadataBlock(metadata_block_header, metadata_block_stream_info)
    
    metadata_blocks = (metadata_block, )
    
    # Creo il nuovo flusso
    stream = Stream(metadata_blocks, frames)

    return stream

def make_subframe_constant(channel, sample_index):
    # Frame
    signal = channel[sample_index : sample_index + BLOCK_SIZE]
    # Primo campione
    first_sample = signal[0]

    # Se tutti i campioni sono uguali 
    for sample in signal:
        if sample != first_sample:
            return None
    
    # Costruisci una subframe constant
    return SubframeConstant(first_sample)

def make_subframe_verbatim(channel, sample_index):
    signal = channel[sample_index : sample_index + BLOCK_SIZE]

    return SubframeVerbatim(signal)

def fixed_predictor_residual_signal(signal, order):
    # Array di funzioni di predict (lo uso per calcolare le previsioni)
    predictors = [
        lambda signal, index: 0,
        lambda signal, index:     signal[index-1],
        lambda signal, index: 2 * signal[index-1] -     signal[index-2],
        lambda signal, index: 3 * signal[index-1] - 3 * signal[index-2] +     signal[index-3],
        lambda signal, index: 4 * signal[index-1] - 6 * signal[index-2] + 4 * signal[index-3] - signal[index-4],
    ]
    
    residual_signal = list()

    # Predizione dei campioni e store delle differenze
    for index, sample in enumerate(signal[order:], start=order):
        predicted_sample = predictors[order](signal, index)
        residual_sample = sample - predicted_sample

        residual_signal.append(residual_sample)
    
    # Return del residual signal (mantengo solo l'errore)
    return residual_signal

def rice_parameter(residual_signal):
    # Media dei valori assoluti del residual signal
    e_x = math.ceil(sum(map(abs, residual_signal))/len(residual_signal))
    # Logaritmo naturale di 2
    ln_2 = math.log(2)

    # Calcolo del rice parameter
    return math.ceil(math.log2(ln_2 * e_x)) if e_x > 0.0 else 0

def make_subframe_fixed(channel, sample_index, predictor_order):
    # Frame
    signal = channel[sample_index : sample_index + BLOCK_SIZE]
    # Campioni che non passo al predittore
    warmup_samples = channel[sample_index : sample_index + predictor_order]

    if len(signal) <= predictor_order or len(warmup_samples) < predictor_order:
        return None

    # Calcolo del residual signal
    residual_signal = fixed_predictor_residual_signal(signal, predictor_order)
    # Calcolo del parametro rice
    parameter = rice_parameter(residual_signal)

    partition_order = 0
    rice_partition = (Rice2Partition(parameter, residual_signal),)
    partitioned_rice = PartitionedRice(partition_order, rice_partition)
    residual = Residual(RESIDUAL_CODING_METHOD_PARTITION_RICE2, partitioned_rice)

    return SubframeFixed(predictor_order, warmup_samples, residual)

def write_stream(stream, output_path):
    with open(output_path, 'wb') as output_file:
        output_file.write(stream.get_bytes())

class WaveStream:
    def __init__(self, sample_size, sample_rate, channels, md5_digest):
        self.sample_size = sample_size   
        self.sample_rate = sample_rate      
        self.channels = channels            
        self.num_channels = len(channels)
        self.num_samples = len(channels[0])
        self.md5_digest = md5_digest

if __name__ == "__main__":
    main(sys.argv)