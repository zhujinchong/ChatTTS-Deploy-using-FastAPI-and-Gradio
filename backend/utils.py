import re
import numpy as np


def normalize_audio(audio):
    """
    Normalize audio array to be between -1 and 1
    :param audio: Input audio array
    :return: Normalized audio array
    """
    audio = np.clip(audio, -1, 1)
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    return audio


def combine_audio(wavs):
    """
    合并多段音频
    :param wavs:
    :return:
    """
    wavs = [normalize_audio(w) for w in wavs]  # 先对每段音频归一化
    # wavs = np.array(wavs, dtype=np.float32)
    # return wavs
    combined_audio = np.concatenate(wavs, axis=1)  # 沿着时间轴合并
    return normalize_audio(combined_audio)  # 合并后再次归一化


def remove_chinese_punctuation(text):
    """
    移除文本中的中文标点符号 [：；！（），【】『』「」《》－‘“’”:,;!\(\)\[\]><\-] 替换为 ，
    :param text:
    :return:
    """
    chinese_punctuation_pattern = r"[：；！（），【】『』「」《》－‘“’”:,;!\(\)\[\]><\-·]"
    text = re.sub(chinese_punctuation_pattern, '，', text)
    # 使用正则表达式将多个连续的句号替换为一个句号
    text = re.sub(r'[。，]{2,}', '。', text)
    # 删除开头和结尾的 ， 号
    text = re.sub(r'^，|，$', '', text)
    return text


def remove_english_punctuation(text):
    """
    移除文本中的中文标点符号 [：；！（），【】『』「」《》－‘“’”:,;!\(\)\[\]><\-] 替换为 ，
    :param text:
    :return:
    """
    chinese_punctuation_pattern = r"[：；！（），【】『』「」《》－‘“’”:,;!\(\)\[\]><\-·]"
    text = re.sub(chinese_punctuation_pattern, ',', text)
    # 使用正则表达式将多个连续的句号替换为一个句号
    text = re.sub(r'[,\.]{2,}', '.', text)
    # 删除开头和结尾的 ， 号
    text = re.sub(r'^,|,$', '', text)
    return text


def detect_language(sentence):
    # ref: https://github.com/2noise/ChatTTS/blob/main/ChatTTS/utils/infer_utils.py#L55
    chinese_char_pattern = re.compile(r'[\u4e00-\u9fff]')
    english_word_pattern = re.compile(r'\b[A-Za-z]+\b')

    chinese_chars = chinese_char_pattern.findall(sentence)
    english_words = english_word_pattern.findall(sentence)

    if len(chinese_chars) > len(english_words):
        return "zh"
    else:
        return "en"


def split_text(text, min_length=60):
    """
    将文本分割为长度不小于min_length的句子
    """
    # 短句分割符号
    sentence_delimiters = re.compile(r'([。？！\.]+)')
    # 匹配多个连续的回车符 作为段落点 强制分段
    paragraph_delimiters = re.compile(r'(\s*\n\s*)+')

    paragraphs = re.split(paragraph_delimiters, text)

    result = []

    for paragraph in paragraphs:
        if not paragraph.strip():
            continue  # 跳过空段落
        # 小于阈值的段落直接分开
        if len(paragraph.strip()) < min_length:
            result.append(paragraph.strip())
            continue
        # 大于的再计算拆分
        sentences = re.split(sentence_delimiters, paragraph)
        current_sentence = ''
        for sentence in sentences:
            if re.match(sentence_delimiters, sentence):
                current_sentence += sentence.strip() + ''
                if len(current_sentence) >= min_length:
                    result.append(current_sentence.strip())
                    current_sentence = ''
            else:
                current_sentence += sentence.strip()

        if current_sentence:
            if len(current_sentence) < min_length and len(result) > 0:
                result[-1] += current_sentence
            else:
                result.append(current_sentence)
    if detect_language(text[:1024]) == "zh":
        result = [normalize_zh(_.strip()) for _ in result if _.strip()]
    else:
        result = [normalize_en(_.strip()) for _ in result if _.strip()]
    return result


def normalize_en(text):
    # 不再在 ChatTTS 外正则化文本
    # from tn.english.normalizer import Normalizer
    # normalizer = Normalizer()
    # text = normalizer.normalize(text)
    text = remove_english_punctuation(text)
    return text


def normalize_zh(text):
    # 不再在 ChatTTS 外正则化文本
    # from tn.chinese.normalizer import Normalizer
    # normalizer = Normalizer()
    # text = normalizer.normalize(text)
    text = remove_chinese_punctuation(text)
    return text


def batch_split(items, batch_size=5):
    """
    将items划分为大小为batch_size的批次
    :param items:
    :param batch_size:
    :return:
    """
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


# if __name__ == '__main__':
#     text = """
# hello
# """
#     print(split_text(text))