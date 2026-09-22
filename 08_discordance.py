def detect_discordance(note, sentiment):
    if note is None or sentiment is None:
        return False
    if note >= 4 and sentiment == "negative":
        return True
    if note <= 2 and sentiment == "positive":
        return True
    return False
