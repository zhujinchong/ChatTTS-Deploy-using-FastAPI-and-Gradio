# **一、项目简介**

ChatTTS优点：

* 支持中英文
* 支持流式返回

ChatTTS缺点：

* 还是有点慢
* 因为训练数据中添加了少量高频噪声，所以合成语音感觉不稳定。偶尔有个字读不出来。

使用 FastAPI 和 Gradio 本地部署 ChatTTS 文本转语音模型，并通过 Docker Compose 进行容器化部署。

**操作流程demo：**

# **二、部署**

## 公共步骤

先克隆本项目：

```
git clone git@github.com:zhujinchong/ChatTTS-Deploy-using-FastAPI-and-Gradio.git
cd ChatTTS-Deploy-using-FastAPI-and-Gradio
```

再下载ChatTTS模型：

```
cd backend/models/
# git lfs install
git lfs clone https://www.modelscope.cn/mirror013/ChatTTS.git
# git lfs clone https://www.modelscope.cn/AI-ModelScope/ChatTTS.git 新版模型不能用
```

## 方式一：本地部署

安装环境依赖：

```
conda create -n tts python==3.9
conda activate tts 
pip install --upgrade pip
cd backend
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

运行后端：

```
cd backend/fastapi
python api.py
```



## 方式二：Docker部署

运行

```
docker compose build
docker compose up
```

这个命令将会：

* 构建FastAPI和Streamlit服务的Docker镜像。
* 启动两个服务，将FastAPI暴露在9880端口，Streamlit暴露在7860端口。

# **三、ChatTTS参数**

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

# **四、问题**

**问题1：中文日期(2023.12)或数字(1,2,3)等无法识别。**

参考：[中文部分无法识别数字（1，2，3等）和句号（。）是需要设置什么吗？ · Issue #644 · 2noise/ChatTTS (github.com)](https://github.com/2noise/ChatTTS/issues/644)

ChatTTS中代码已经更新，需要安装

```
pynini==2.1.6
WeTextProcessing==1.0.4.1
```

原理：换成中文大写数字/日期（二零二三年十二月）。

已解决。(chat.infer已经内置，参数 `do_text_normalization=True`)

**问题2：中文标点符号误识别。**

已解决。(utils中正则替换)

**问题3：长文本需要切分成短的。**

已解决。(utils中切分成批处理)

**问题4：支持流式返回。**

待解决

**问题5：Gradio前端支持。**

待解决

**问题：ChatTTS用的哪个版本的代码？**

参考ChatTTS_colab，操作如下：

```
cd backend
git clone https://github.com/2noise/ChatTTS
cd ChatTTS
git checkout e6412b1
cd ..
mv ChatTTS temp
mv temp/ChatTTS ./ChatTTS
rm -rf temp
```

# **参考**

- https://github.com/2noise/ChatTTS
- https://github.com/yuanquderzi/ChatTTS-Deployment-using-FastAPI-and-Streamlit
- https://github.com/6drf21e/ChatTTS_colab
- https://blog.csdn.net/u010522887/article/details/139719895
