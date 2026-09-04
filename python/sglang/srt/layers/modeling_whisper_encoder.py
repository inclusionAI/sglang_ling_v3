import logging

import torch.nn.functional as F
from torch import Tensor
from whisper.model import AudioEncoder

logger = logging.getLogger(__name__)


class WhisperAudioEncoder(AudioEncoder):
    """
    We inherited the original Whisper encoder and modified its 30-second fixed-length padding logic to
    improve training and inference efficiency.
    """

    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__(n_mels, n_ctx, n_state, n_head, n_layer)
        self.audio_emb_dim = n_state

    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = [B, T, d] the mel spectrogram of the audio
        """
        x = x.transpose(1, 2)
        x = F.gelu(self.conv1(x))  # kernel_size=3, stride=1, padding=1
        x = F.gelu(self.conv2(x))  # kernel_size=3, stride=2, padding=1
        x = x.permute(0, 2, 1)
        positional_embedding = self.positional_embedding[: x.shape[1], :]
        assert x.shape[1:] == positional_embedding.shape, "incorrect audio shape"
        x = (x + positional_embedding).to(x.dtype)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        return x
