#!/usr/bin/env python3
import ChatTTS
import wave
import numpy as np
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import StreamingResponse, JSONResponse, Response
from starlette.middleware.cors import CORSMiddleware  # 引入 CORS中间件模块
import uvicorn
import soundfile
import io
import typing
import pydantic
import torch


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # "*"，即为所有。
    allow_credentials=True,
    allow_methods=["*"],  # 设置允许跨域的http方法，比如 get、post、put等。
    allow_headers=["*"],
)  # 允许跨域的headers，可以用来鉴别来源等作用。


class TTSInput(pydantic.BaseModel):
    text: str = None
    seed: int = 697
    speed: int = 3
    streaming: bool = False


def get_chat_model() -> ChatTTS.Chat:
    chat = ChatTTS.Chat()
    chat.load_models(source="custom", local_path="./models/ChatTTS")
    return chat


def clear_cuda_cache():
    """
    Clear CUDA cache
    """
    torch.cuda.empty_cache()


def deterministic(seed=10):
    """
    Set random seed for reproducibility
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_tts_audio(request: TTSInput, chat: ChatTTS.Chat):
    deterministic(seed=request.seed)
    # 选择speaker：https://modelscope.cn/studios/ttwwwaa/ChatTTS_Speaker
    r = chat.sample_random_speaker()
    # 推理参数
    params_infer_code = {
        "spk_emb": r,  # add sampled speaker
        "prompt": f"[speed_{request.speed}]",
        "temperature": 0.1,  # using customtemperature
        "top_P": 0.7,  # top P decode
        "top_K": 1,  # top K decode
    }
    # refine参数：口语/笑/停顿
    params_refine_text = {"prompt": "[oral_2][laugh_0][break_6]"}

    # 推理
    if request.streaming:
        wavs_gen = chat.infer(
            [request.text],
            skip_refine_text=False,  #
            refine_text_only=True,  # 模型自动添加停顿
            params_infer_code=params_infer_code,
            params_refine_text=params_refine_text,
            use_decoder=True,
            stream=True,
        )
        for gen in wavs_gen:
            wavs = [np.array([[]])]
            wavs[0] = np.hstack([wavs[0], np.array(gen[0])])
            audio_data = np.array(wavs[0], dtype=np.float32)
            audio_data = (audio_data * 32767).astype(np.int16)
            yield audio_data
    else:
        wavs = chat.infer(
            [request.text],
            skip_refine_text=False,  #
            refine_text_only=True,  # 模型自动添加停顿
            params_infer_code=params_infer_code,
            params_refine_text=params_refine_text,
            use_decoder=True,
        )
        clear_cuda_cache()
        audio_data = wavs[0]
        audio_data = np.array(wavs[0], dtype=np.float32)
        audio_data = (audio_data * 32767).astype(
            np.int16
        )  # 将浮点数音频样本缩放到 int16 类型能表示的范围内
        yield audio_data


@app.post("/tts")
def tts(request: TTSInput, chat: ChatTTS.Chat = Depends(get_chat_model)):
    try:
        sample_rate = 24000
        if request.streaming:
            tts_generator = generate_tts_audio(request)

            def streaming_generator(tts_generator: typing.Generator):
                is_first = True
                if is_first:
                    # This will create a wave header then append the frame input
                    # It should be first on a streaming wav file
                    # Other frames better should not have it (else you will hear some artifacts each chunk start)
                    wav_buf = io.BytesIO()
                    with wave.open(wav_buf, "wb") as vfout:
                        vfout.setnchannels(1)
                        vfout.setsampwidth(2)
                        vfout.setframerate(sample_rate)
                        vfout.writeframes(b"")
                    wav_buf.seek(0)
                    is_first = False
                    yield wav_buf.read()
                else:
                    for chunk in tts_generator:
                        io_buffer = io.BytesIO()
                        soundfile.write(io_buffer, chunk.tobytes(), sample_rate, "wav")
                        io_buffer.seek(0)
                        res_audio_data = io_buffer.getvalue()
                        yield res_audio_data

            return StreamingResponse(
                streaming_generator(tts_generator), media_type="audio/wav"
            )

        else:
            audio_data = next(generate_tts_audio(request))
            io_buffer = io.BytesIO()
            soundfile.write(io_buffer, audio_data.tobytes(), sample_rate, "wav")
            io_buffer.seek(0)
            res_audio_data = io_buffer.getvalue()
            return Response(res_audio_data, media_type="audio/wav")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9880, workers=1)
