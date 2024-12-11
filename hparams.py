from dataclasses import dataclass


@dataclass
class HParams:
	"""
	simple data class that houses all of the hyperparameters
	"""
	n_vocab: int                 # size of text vocabulary
	symbol_embedding_dim: int    # dimension of character/phoneme embedding
	hidden_dim: int              # dimension of hidden layers for encoder/decoder/converter
	speaker_embedding_dim: int   # dimension of speaker embedding (if multi-speaker)
	n_speakers: int              # number of speakers (1 if single-speaker)
	dropout: float               # dropout probability
	encoder_n_conv: int          # number of encoder convolution blocks
	decoder_n_conv: int          # number of decoder convolution/attention blocks
	converter_n_conv: int        # number of converter convolution blocks
	mel_dim: int                 # dimension of mel spectrogram frames
	linear_dim: int              # dimension of linear spectrogram frames
	reduction_factor: int        # number of mel frames predicted at once by the decoder
	monotonic_attention: bool    # whether to use monotonic attention (optional)
	position_rate: float         # position rate for positional encoding
	kernel_size: int 			 # kernel size of convolutional layer. must be odd