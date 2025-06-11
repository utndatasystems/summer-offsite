from string import printable
import numpy as np
import random

class ArithmeticCoderBase(object):
    # Constructs an arithmetic coder, which initializes the code range.
    def __init__(self, statesize):
        #if statesize < 1:
            #raise ValueError("State size out of range")
        # -- Configuration fields --
        # Number of bits for the 'low' and 'high' state variables. Must be at least 1.
        # - Larger values are generally better - they allow a larger maximum frequency total (MAX_TOTAL),
        #   and they reduce the approximation error inherent in adapting fractions to integers;
        #   both effects reduce the data encoding loss and asymptotically approach the efficiency
        #   of arithmetic coding using exact fractions.
        # - But larger state sizes increase the computation time for integer arithmetic,
        #   and compression gains beyond ~30 bits essentially zero in real-world applications.
        # - Python has native bigint arithmetic, so there is no upper limit to the state size.
        #   For Java and C++ where using native machine-sized integers makes the most sense,
        #   they have a recommended value of STATE_SIZE=32 as the most versatile setting.
        self.STATE_SIZE = statesize
        # Maximum range (high+1-low) during coding (trivial), which is 2^STATE_SIZE = 1000...000.
        self.MAX_RANGE = 1 << self.STATE_SIZE
        # Minimum range (high+1-low) during coding (non-trivial), which is 0010...010.
        self.MIN_RANGE = (self.MAX_RANGE >> 2) + 2
        # Maximum allowed total from a frequency table at all times during coding. This differs from Java
        # and C++ because Python's native bigint avoids constraining the size of intermediate computations.
        self.MAX_TOTAL = self.MIN_RANGE
        # Bit mask of STATE_SIZE ones, which is 0111...111.
        self.MASK = self.MAX_RANGE - 1
        # The top bit at width STATE_SIZE, which is 0100...000.
        self.TOP_MASK = self.MAX_RANGE >> 1
        # The second highest bit at width STATE_SIZE, which is 0010...000. This is zero when STATE_SIZE=1.
        self.SECOND_MASK = self.TOP_MASK >> 1

        # -- State fields --
        # Low end of this arithmetic coder's current range. Conceptually has an infinite number of trailing 0s.
        self.low = 0
        # High end of this arithmetic coder's current range. Conceptually has an infinite number of trailing 1s.
        self.high = self.MASK
#         print("STATE_SIZE  : ",self.STATE_SIZE)
#         print("MAX_RANGE   : ",bin(self.MAX_RANGE))
#         print("MIN_RANGE   : ",bin(self.MIN_RANGE))
#         print("MAX_TOTAL   : ",bin(self.MAX_TOTAL))
#         print("MASK        : ",bin(self.MASK))
#         print("TOP_MASK    : ",bin(self.TOP_MASK))
#         print("SECOND_MASK : ",bin(self.SECOND_MASK))
#         print("low         : ",bin(self.low))
#         print("high        : ",bin(self.high))


    # Updates the code range (low and high) of this arithmetic coder as a result
    # of processing the given symbol with the given frequency table.
    # Invariants that are true before and after encoding/decoding each symbol:
    # - 0 <= low <= code <= high < 2^STATE_SIZE. ('code' exists only in the decoder.)
    #   Therefore these variables are unsigned integers of STATE_SIZE bits.
    # - (low < 1/2 * 2^STATE_SIZE) && (high >= 1/2 * 2^STATE_SIZE).
    #   In other words, they are in different halves of the full range.
    # - (low < 1/4 * 2^STATE_SIZE) || (high >= 3/4 * 2^STATE_SIZE).
    #   In other words, they are not both in the middle two quarters.
    # - Let range = high - low + 1, then MAX_RANGE/4 < MIN_RANGE <= range
    #   <= MAX_RANGE = 2^STATE_SIZE. These invariants for 'range' essentially
    #   dictate the maximum total that the incoming frequency table can have.
    def update(self,  cumul, symbol):
        # State check
        low = self.low
        high = self.high
        #if low >= high or (low & self.MASK) != low or (high & self.MASK) != high:
            #raise AssertionError("Low or high out of range")
        range = high - low + 1
        #if not (self.MIN_RANGE <= range <= self.MAX_RANGE):
            #raise AssertionError("Range out of range")

        # Frequency table values check
        total = cumul[-1].item()
        symlow = cumul[symbol].item()
        symhigh = cumul[symbol+1].item()
        #if symlow == symhigh:
            #raise ValueError("Symbol has zero frequency")
        #if total > self.MAX_TOTAL:
            #raise ValueError("Cannot code symbol because total is too large")

        # Update range
        newlow  = low + symlow  * range // total
        newhigh = low + symhigh * range // total - 1
        self.low = newlow
        self.high = newhigh
        # While the highest bits are equal
#         print("New loop")
#         print(bin(self.low),"; ",bin(self.high))
#         print((self.low ^ self.high) & self.TOP_MASK)
        while ((self.low ^ self.high) & self.TOP_MASK) == 0:
            self.shift()
#             print("After shift:",bin(self.low),"; ",bin(self.high))
            self.low = (self.low << 1) & self.MASK
            self.high = ((self.high << 1) & self.MASK) | 1
#             print(bin(self.low),"; ",bin(self.high))

        # While the second highest bit of low is 1 and the second highest bit of high is 0
#         print(self.low & ~self.high & self.SECOND_MASK)
            
        while (self.low & ~self.high & self.SECOND_MASK) != 0:
            self.underflow()
#             print("After underflow",bin(self.low),"; ",bin(self.high))
            self.low = (self.low << 1) & (self.MASK >> 1)
            self.high = ((self.high << 1) & (self.MASK >> 1)) | self.TOP_MASK | 1
#             print(bin(self.low),"; ",bin(self.high))


    # Called to handle the situation when the top bit of 'low' and 'high' are equal.
    def shift(self):
        raise NotImplementedError()


    # Called to handle the situation when low=01(...) and high=10(...).
    def underflow(self):
        raise NotImplementedError()



# Encodes symbols and writes to an arithmetic-coded bit stream.
class ArithmeticEncoder(ArithmeticCoderBase):

    # Constructs an arithmetic coding encoder based on the given bit output stream.
    def __init__(self, statesize, bitout):
        super(ArithmeticEncoder, self).__init__(statesize)
        # The underlying bit output stream.
        self.output = bitout
        # Number of saved underflow bits. This value can grow without bound.
        self.num_underflow = 0


    # Encodes the given symbol based on the given frequency table.
    # This updates this arithmetic coder's state and may write out some bits.
    def write(self, cumul, symbol):
    #		if not isinstance(freqs, CheckedFrequencyTable):
    #			freqs = CheckedFrequencyTable(freqs)
        self.update(cumul, symbol)


    # Terminates the arithmetic coding by flushing any buffered bits, so that the output can be decoded properly.
    # It is important that this method must be called at the end of the each encoding process.
    # Note that this method merely writes data to the underlying output stream but does not close it.
    def finish(self):
        self.output.write(1)


    def shift(self):
        bit = self.low >> (self.STATE_SIZE - 1)
        self.output.write(bit)

        # Write out the saved underflow bits
        for _ in range(self.num_underflow):
            self.output.write(bit ^ 1)
        self.num_underflow = 0


    def underflow(self):
        self.num_underflow += 1



# Reads from an arithmetic-coded bit stream and decodes symbols.
class ArithmeticDecoder(ArithmeticCoderBase):

    # Constructs an arithmetic coding decoder based on the
    # given bit input stream, and fills the code bits.
    def __init__(self, statesize, bitin):
        super(ArithmeticDecoder, self).__init__(statesize)
        # The underlying bit input stream.
        self.input = bitin
        # The current raw code bits being buffered, which is always in the range [low, high].
        self.code = 0
        for _ in range(self.STATE_SIZE):
            self.code = self.code << 1 | self.read_code_bit()


    # Decodes the next symbol based on the given frequency table and returns it.
    # Also updates this arithmetic coder's state and may read in some bits.
    def read(self, cumul, alphabet_size):
    #		if not isinstance(freqs, CheckedFrequencyTable):
    #			freqs = CheckedFrequencyTable(freqs)

        # Translate from coding range scale to frequency table scale
        total = cumul[-1].item()
    #		if total > self.MAX_TOTAL:
    #			raise ValueError("Cannot decode symbol because total is too large")
        range = self.high - self.low + 1
        offset = self.code - self.low
        value = ((offset + 1) * total - 1) // range
    #		assert value * range // total <= offset
    #		assert 0 <= value < total

        # A kind of binary search. Find highest symbol such that freqs.get_low(symbol) <= value.
        start = 0
        end = alphabet_size
        while end - start > 1:
            middle = (start + end) >> 1
            if cumul[middle] > value:
                end = middle
            else:
                start = middle
    #		assert start + 1 == end

        symbol = start
    #		assert freqs.get_low(symbol) * range // total <= offset < freqs.get_high(symbol) * range // total
        self.update(cumul, symbol)
    #		if not (self.low <= self.code <= self.high):
    #			raise AssertionError("Code out of range")
        return symbol


    def shift(self):
        self.code = ((self.code << 1) & self.MASK) | self.read_code_bit()


    def underflow(self):
        self.code = (self.code & self.TOP_MASK) | ((self.code << 1) & (self.MASK >> 1)) | self.read_code_bit()


    # Returns the next bit (0 or 1) from the input stream. The end
    # of stream is treated as an infinite number of trailing zeros.
    def read_code_bit(self):
        temp = self.input.read()
        if temp == -1:
            temp = 0
        return temp


# ------------------------------------------------------------------
# Minimal bit-stream helpers
# ------------------------------------------------------------------
class BitOutputStream:
    """Collects single bits written by the encoder."""
    def __init__(self):
        self.bits = []

    def write(self, bit: int):
        self.bits.append(bit & 1)  # store only the LSB

    def get_bits(self):
        return self.bits


class BitInputStream:
    """Feeds bits (and infinite trailing zeros) to the decoder."""
    def __init__(self, bits):
        self.bits = bits
        self.index = 0

    def read(self):
        if self.index < len(self.bits):
            b = self.bits[self.index]
            self.index += 1
            return b
        else:                     # spec: return –1 → decoder treats as 0
            return -1


# ------------------------------------------------------------------
# Utility – convert probabilities → integer cumulative vector
# ------------------------------------------------------------------
def build_cumul(prob_vec: np.ndarray, total: int = 4096) -> np.ndarray:
    """
    Turn a length-N probability vector that sums to 1 into the length-(N+1)
    cumulative-frequency array expected by ArithmeticCoder.  Ensures every
    symbol’s frequency ≥ 1 and that the overall total == `total`.
    """
    alphabet_size = prob_vec.size

    # Give every symbol at least one count so no zero-frequency symbols
    freq = np.maximum(1,
                      (prob_vec * (total - alphabet_size)).astype(np.int64))
    freq += 1
    diff = freq.sum() - total

    # Adjust so the sum is exact
    if diff > 0:                        # too many counts → take some back
        for idx in np.argsort(-freq):   # biggest bars first
            take = min(freq[idx] - 1, diff)
            freq[idx] -= take
            diff -= take
            if diff == 0:
                break
    elif diff < 0:                      # not enough counts → add
        for idx in np.argsort(-prob_vec):
            freq[idx] += 1
            diff += 1
            if diff == 0:
                break

    assert freq.sum() == total and np.all(freq >= 1)

    cumul = np.empty(alphabet_size + 1, dtype=np.int64)
    cumul[0] = 0
    cumul[1:] = np.cumsum(freq)
    return cumul

class LLMCompressor:
    def __init__(self):
        self.bitout = BitOutputStream()
        self.encoder = ArithmeticEncoder(32, self.bitout)

    def next_token(self, correct_token_idx, probs):
        self.encoder.write(build_cumul(probs), correct_token_idx)

    def compress(self):
        self.encoder.finish()
        return self.bitout.get_bits()

class LLMDecompressor:
    def __init__(self, code):
        self.decoder = ArithmeticDecoder(32, BitInputStream(code))

    def decompress(self, probs) -> int: # Returns one single token at a time (returns the index of the token)
        cumul = build_cumul(probs)
        return self.decoder.read(cumul, len(probs))

def test_compress_decompress():
    alphabet = printable
    alphabet_size = len(alphabet)
    text_len = 1000
    text = [random.randint(0, len(alphabet)-1) for _ in range(text_len)]
    prob_tables = []
    for _ in range(text_len):
        probs = np.random.rand(alphabet_size)
        prob_tables.append(probs / probs.sum()) # normalise to 1

    compressor = LLMCompressor()
    for i in range(text_len):
        correct_token_idx = text[i]
        probs = prob_tables[i]
        compressor.next_token(correct_token_idx, probs)
    code = compressor.compress()

    decompressor = LLMDecompressor(code)
    for i in range(text_len):
        probs = prob_tables[i]
        token_idx = decompressor.decompress(probs)
        assert token_idx == text[i], f"Decompressed token {token_idx} does not match original {text[i]} at index {i}"
    print("Decompression successful, all tokens match the original text.")
