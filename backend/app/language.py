def detect_output_language(*values: object) -> str:
    text = " ".join(str(value) for value in values if value)
    chinese_chars = sum("\u4e00" <= char <= "\u9fff" for char in text)
    alpha_chars = sum(char.isalpha() for char in text)

    if chinese_chars >= 6 or (alpha_chars and chinese_chars / alpha_chars >= 0.2):
        return "Chinese"

    return "English"


def language_instruction(language: str) -> str:
    if language == "Chinese":
        return "请使用中文输出，保留必要的英文技术名词。"

    return "Write in English. Preserve technical terms as written by the user."
