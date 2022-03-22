import unidecode
import string
import sklearn

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


def normalize_text(text: str) -> str:
    """normalize or remove all the accents (diacritics)
    :param text: input text
    :return: normalized text

    Example: 'Málaga' -> 'Malaga'
    """
    normalized_text = unidecode.unidecode(text)
    return normalized_text


def compute_roc_auc(y_prob, y):
    """compute false positive rate, true positive rate, and auc score
    :param y_prob: predicted probability
    :param y: label
    """
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y, y_prob)
    auc_score = sklearn.metrics.auc(fpr, tpr)
    return fpr, tpr, auc_score


def compute_score(y_pred, y):
    """compute accuracy and f1-score
    :param y_prob: prediction
    :param y: label
    """
    acc = sklearn.metrics.accuracy_score(y, y_pred)
    f1 = sklearn.metrics.f1_score(y, y_pred)
    return acc, f1
