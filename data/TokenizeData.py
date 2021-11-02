from torchnlp.encoders.text import StaticTokenizerEncoder, pad_tensor


class Tokenize:
    def __init__(self, _train_captions, _max_length, _cfg):
        self.tokenizer = lambda s: s.split()
        self.max_length = _max_length
        self.min_occur = _cfg["MIN_OCCURRENCES"]
        self._encode_data(_train_captions)
        self.vocab_size = self.encoder.vocab_size

    def _encode_data(self, train_captions):
        self.encoder = StaticTokenizerEncoder(train_captions,
                                              tokenize=self.tokenizer,
                                              min_occurrences=self.min_occur,
                                              padding_index=0,
                                              unknown_index=1)

        encoded_data = [self.encoder.encode(e) for e in train_captions]
        self.encoded_data = [pad_tensor(x, length=self.max_length) for x in encoded_data]
