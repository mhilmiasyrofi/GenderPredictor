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

