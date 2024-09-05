#!/usr/bin/env python3
import ChatTTS
import numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse, Response
from starlette.middleware.cors import CORSMiddleware  # 引入 CORS中间件模块
from contextlib import asynccontextmanager
import uvicorn
import soundfile
import io
from pydantic import BaseModel, Field
from typing import List, Generator
import torch
import json
from utils import split_text, batch_split, combine_audio_to_int
import wave


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load resources
    global chat
    chat = ChatTTS.Chat()
    chat.load_models(source="custom", custom_path="./models/ChatTTS")

    global speaker_voice
    with open('./speaker_voice.json', 'r', encoding='utf-8') as json_file:
            speaker_voice = json.load(json_file)
    yield
    # release resources
    pass


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # "*"，即为所有。
    allow_credentials=True,
    allow_methods=["*"],  # 设置允许跨域的http方法，比如 get、post、put等。
    allow_headers=["*"],
)  # 允许跨域的headers，可以用来鉴别来源等作用。


class TTSInput(BaseModel):
    text: str = Field(..., description="The text to be converted to speech")
    speed: int = Field(default=3, ge=0, le=9, description="The speed of the speech")
    speaker_id: int = Field(default=1, ge=1, le=22, description="The speaker id")
    speaker_emb: List[float] = Field(default=None, description="The speaker embedding vector")


def clear_cuda_cache():
    """
    Clear CUDA cache
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def deterministic(seed=10):
    """
    Set random seed for reproducibility
    """
    if torch.cuda.is_available():
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@app.post("/tts")
def tts(request: TTSInput):
    # ========================参数=================================================
    texts = split_text(request.text)
    print(texts)
    # 选择speaker：https://modelscope.cn/studios/ttwwwaa/ChatTTS_Speaker
    # rnd_spk_emb = torch.tensor([float(x) for x in request.speaker.split(',')])
    if request.speaker_emb:
        rnd_spk_emb = torch.tensor(request.speaker_emb)
    else:
        rnd_spk_emb = torch.tensor(speaker_voice[str(request.speaker_id)]['tensor'])
    # 推理参数
    params_infer_code = {
        "spk_emb": rnd_spk_emb,  # add sampled speaker
        "prompt": f"[speed_{request.speed}]",
        "temperature": 0.1,  # using customtemperature
        "top_P": 0.7,  # top P decode
        "top_K": 1,  # top K decode
    }
    # refine参数：口语/笑/停顿
    params_refine_text = {
        "prompt": "[oral_0][laugh_0][break_0]",
    }
    # ========================推理=================================================
    deterministic()
    all_wavs = []
    for batch in batch_split(texts):
        wavs = chat.infer(
            batch,
            params_infer_code=params_infer_code,
            params_refine_text=params_refine_text,
            use_decoder=True,
            stream=False,
        )
        for x in wavs:
            audio_data = x[0]
            # audio_data = audio_data / np.max(np.abs(audio_data))
            all_wavs.append(audio_data)
    clear_cuda_cache()
    # audio = (np.concatenate(all_wavs) * 32767).astype(np.int16)
    audio = combine_audio_to_int(all_wavs)
    # ========================存储/返回=================================================
    # 使用 BytesIO 作为缓冲区来存储 WAV 文件的内容。
    io_buffer = io.BytesIO()
    soundfile.write(io_buffer, audio, samplerate=24000, format="wav")
    io_buffer.seek(0)
    # res_audio_data = io_buffer.getvalue()
    # FileResponse 来返回文件
    return Response(content=io_buffer.getvalue(), media_type="audio/wav")


def generate_tts_audio(request: TTSInput):
    # ========================参数=================================================
    texts = split_text(request.text)
    print(texts)
    # 选择speaker：https://modelscope.cn/studios/ttwwwaa/ChatTTS_Speaker
    # rnd_spk_emb = torch.tensor([float(x) for x in request.speaker.split(',')])
    if request.speaker_emb:
        rnd_spk_emb = torch.tensor(request.speaker_emb)
    else:
        rnd_spk_emb = torch.tensor(speaker_voice[str(request.speaker_id)]['tensor'])
    # 推理参数
    params_infer_code = {
        "spk_emb": rnd_spk_emb,  # add sampled speaker
        "prompt": f"[speed_{request.speed}]",
        "temperature": 0.1,  # using customtemperature
        "top_P": 0.7,  # top P decode
        "top_K": 1,  # top K decode
    }
    # refine参数：口语/笑/停顿
    params_refine_text = {
        "prompt": "[oral_0][laugh_0][break_0]",
    }
    # ========================推理=================================================
    deterministic()
    for text in texts:
        wavs_gen = chat.infer(
            text,
            params_infer_code=params_infer_code,
            params_refine_text=params_refine_text,
            use_decoder=True,
            stream=True,
        )
        for gen in wavs_gen:
            wavs = [np.array([[]])]
            wavs[0] = np.hstack([wavs[0], np.array(gen[0])])
            audio_data = wavs[0][0]
            audio_data = audio_data / np.max(np.abs(audio_data))
            audio = (audio_data * 32767).astype(np.int16)
            yield audio
    clear_cuda_cache()


@app.post("/tts/streaming")
def tts_stream(request: TTSInput):
    tts_generator = generate_tts_audio(request)
    def streaming_generator(tts_generator: Generator):
        # 不知道为什么加这一段，删了可以吗？不可以，删了Gradio那里测得有问题
        wav_buf = io.BytesIO()
        with wave.open(wav_buf, "wb") as vfout:
            vfout.setnchannels(1)
            vfout.setsampwidth(2)
            vfout.setframerate(24000)
            vfout.writeframes(b'')
        wav_buf.seek(0)
        yield wav_buf.read()
        for chunk in tts_generator:
            # yield chunk.tobytes()
            io_buffer = io.BytesIO()
            soundfile.write(io_buffer, chunk, samplerate=24000, format="wav")
            io_buffer.seek(0)
            yield io_buffer.getvalue()
    return StreamingResponse(streaming_generator(tts_generator), media_type="audio/wav")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9880, workers=1)
