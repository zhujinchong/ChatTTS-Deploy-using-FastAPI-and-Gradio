# 使用官方Python运行时作为基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 将当前目录的内容复制到容器中的/streamlit目录
COPY ui.py /app
COPY requirements.txt /app

# 安装requirements.txt中指定的所有依赖包
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 暴露Streamlit的默认端口
EXPOSE 7860

# 运行Streamlit应用
CMD ["python", "ui.py"]
