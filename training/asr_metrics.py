"""Exact-text ASR scoring shared by training and model comparisons."""

import re
import unicodedata


def words(text):
    normalized = unicodedata.normalize("NFC", text).casefold()
    return "".join(" " if unicodedata.category(char)[0] in "PS" else char for char in normalized).split()


def edit_distance(first, second):
    previous = list(range(len(second) + 1))
    for i, left in enumerate(first, start=1):
        current = [i]
        for j, right in enumerate(second, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def score(references, predictions):
    pairs = [(words(reference), words(prediction)) for reference, prediction in zip(references, predictions, strict=True)]

    def rate(select):
        selected = [(select(reference), select(prediction)) for reference, prediction in pairs]
        total = sum(len(reference) for reference, _ in selected)
        errors = sum(edit_distance(reference, prediction) for reference, prediction in selected)
        return 100 * errors / total if total else None

    return {
        "wer_percent": rate(lambda tokens: tokens),
        "latin_word_error_percent": rate(lambda tokens: [token for token in tokens if re.search(r"[a-z]", token)]),
    }
