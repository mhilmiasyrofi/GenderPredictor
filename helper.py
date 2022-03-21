import string

def is_alphabet(text: str) -> bool:
    """check whether the text contain non-alphabetic character
    :param text: input text
    :return: the condition
    """
    if len(text) >= 2 :
        for char in text :
            if char not in string.ascii_lowercase :
                return False
    else :
        if text not in string.ascii_lowercase:
            return False
    return True


def remove_punctuation(text: str) -> str:
    """remove punctuation from the input text
    :param text: input text
    :return: cleaned text
    """
    return text.translate(str.maketrans('', '', string.punctuation))


def remove_number(text: str) -> str:
    """remove number from the input text
    :param text: input text
    :return: cleaned text
    """
    return text.translate(str.maketrans('', '', '0123456789'))
