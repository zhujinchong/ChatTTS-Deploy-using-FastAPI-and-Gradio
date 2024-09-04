import gradio as gr
import requests
import io


def tts(text, speaker_id=1):
    response = requests.post('http://localhost:9880/tts', json={"text": text, "speaker_id": speaker_id})
    if response.status_code == 200:
        audio_data = response.content
        return audio_data
    else:
        return None


def tts_streaming(text, speaker_id=1):
    response = requests.post('http://localhost:9880/tts/streaming', json={"text": text, "speaker_id": speaker_id})
    if response.status_code == 200:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:  # 过滤掉 keep-alive 新块
                yield chunk
    else:
        return None


with gr.Blocks() as demo:
    gr.Markdown("# ChatTTS Webui")
    default_text = "四川美食确实以辣闻名，但也有不辣的选择。\n比如甜水面、赖汤圆、蛋烘糕、叶儿粑等，这些小吃口味温和，甜而不腻，也很受欢迎。"        
    text_input = gr.Textbox(label="Input Text", lines=4, placeholder="Please Input Text...", value=default_text)

    with gr.Column():
        speaker_id = gr.Slider(minimum=1, maximum=22, step=1, value=1, label="speaker_id")

    with gr.Column():
        generate_button = gr.Button("Generate")
        audio_output = gr.Audio(label="Audio")
    
    with gr.Column():
        streaming_generate_button = gr.Button("流式Generate")
        streaming_audio_output = gr.Audio(label="流式Audio", streaming=True)

    generate_button.click(tts, inputs=[text_input, speaker_id], outputs=audio_output)
    streaming_generate_button.click(tts_streaming, inputs=[text_input, speaker_id], outputs=streaming_audio_output)
        
# 运行 Gradio 应用
demo.launch(server_name="localhost", server_port=7860)
