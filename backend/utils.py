import ChatTTS
import pydantic
import torch
import numpy as np


class TTSInput(pydantic.BaseModel):
    text: str = None
    seed: int = 697
    speed: int = 3
    streaming: bool = False


def get_chat_model() -> ChatTTS.Chat:
    chat = ChatTTS.Chat()
    chat.load_models(source="local", local_path="./models/ChatTTS")
    return chat


def clear_cuda_cache():
    """
    Clear CUDA cache
    :return:
    """
    torch.cuda.empty_cache()


def deterministic(seed=10):
    """
    Set random seed for reproducibility
    :param seed:
    :return:
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
