## **一、项目简介**

使用 FastAPI 和 Gradio 本地部署 ChatTTS 文本转语音模型，并通过 Docker Compose 进行容器化部署。

**操作流程demo：**


## **二、本地安装使用**

**环境依赖：**

```bash
cuda12.1   
pip install requirements.txt
```

**程序运行方式：**

- 启动FastAPI：用于 API 接口

```bash
cd fastapi
uvicorn server:app --host "0.0.0.0" --port 8000
```

- 启动Streamlit：用于网页

```bash
cd streamlit
streamlit run ui.py
```

- 访问网页：http://localhost:8501
- 本地使用示例

```bash
curl -X POST -H 'content-type: application/json' -d\
   '{"text":"朋友你好啊，今天天气怎么样 ？", "output_path": "abc.wav", "seed":232}' \
    http://localhost:8000/tts
```

- 参数说明：

  text：要合成的文本

  output_path：合成音频的保存路径

  seed：音色种子，不同的种子会产生不同的音色，默认为 697（测试的一个比较好的音色）
- 运行客户端

```bash
python client.py
```


## **三、Docker 部署**

```
docker compose build
docker compose up
```

这个命令将会：

* 构建FastAPI和Streamlit服务的Docker镜像。
* 启动两个服务，将FastAPI暴露在8000端口，Streamlit暴露在8501端口。


## 四、ChatTTS参数

**固定音色**

在ChatTTS中，控制音色的主要是参数 spk_emb，让我们看下源码：

```
def sample_random_speaker(self, ):
  dim = self.pretrain_models['gpt'].gpt.layers[0].mlp.gate_proj.in_features
  # dim=768
  std, mean = self.pretrain_models['spk_stat'].chunk(2)
  # std.size:(768),mean.size(768)
  return torch.randn(dim, device=std.device) * std + mean
```

`sample_random_speaker` 返回的是一个 768 维的向量，也就是说如果你把这个 768 维的向量固定住，就可以获取一个固定的音色了。

音色seed可以参考：[ChatTTS 稳定音色/区分男女 · 创空间 (modelscope.cn)](https://modelscope.cn/studios/ttwwwaa/ChatTTS_Speaker)


**固定语速**

ChatTTS 中，语速是 speed 参数实现的，设置播放的语速，从慢到快，共有10个等级[speed_0]~[speed_9]。

```
params_infer_code = {'prompt':'[speed_2]'}
```


**添加停顿词**

ChatTTS 中，主要的停顿有三种，分别是

* 停顿：停顿词主要有10个等级 [break_0]~[break_9]。当然，模型会根据文本内容自行添加停顿，你也可以在文本中手动添加停顿 [uv_break]。
* 笑声：笑声主要有10个等级[laugh_0]~[laugh_9]。当然，模型会根据文本自动添加笑声，也可以像上面的示例一样手动添加 [laugh].
* 口头语：口头语主要有10个等级[oral_0]~[oral_9]。


## **参考**

- https://github.com/zhujinchong/ChatTTS-Deployment-using-FastAPI-and-Streamlit
- https://github.com/6drf21e/ChatTTS_colab
- https://blog.csdn.net/u010522887/article/details/139719895
