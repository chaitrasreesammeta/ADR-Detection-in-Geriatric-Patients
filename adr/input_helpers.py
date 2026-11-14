MIN_PATIENT_ID_DIGITS = 3
MAX_PATIENT_ID_DIGITS = 5

PUNCTUATION = "?.,!:;#()"

PATIENT_WORDS = [
    "patient",
    "pt",
]

DOSE_UNITS = [
    "mg",
    "mcg",
    "g",
    "ml",
    "units",
]


def extract_number(text: str) -> float | None:
    """
    Pull the first number out of the text
    Returns None if the text has no number.
    """
    number_text = ""

    for character in text:
        if character.isdigit():
            number_text = number_text + character

        elif character == "." and number_text != "" and "." not in number_text:
            number_text = number_text + character

        elif number_text != "":
            break

    if number_text == "":
        return None

    return float(number_text)


def ask_for_number(prompt: str) -> float:
    """
    Ask the user for a number, and keep asking until they type one.
    Words around the number are ignored, e.g. "58 kg" -> 58.0
    """
    while True:
        typed_text = input(prompt).strip()

        number = extract_number(text=typed_text)

        if number is not None:
            return number

        print("  Please enter a number, e.g. 10")


def clean_words(user_input: str) -> list[str]:
    """
    Split the message into lowercase words without punctuation, e.g. "Patient 1210?" -> ["patient", "1210"].
    """
    words = []

    for word in user_input.lower().split():
        cleaned_word = word.strip(PUNCTUATION)
        words.append(cleaned_word)

    return words


def is_patient_id(word: str) -> bool:
    """
    Check if a word is a number with 3 to 5 digits, like "1210".
    """
    if not word.isdigit():
        return False

    if len(word) < MIN_PATIENT_ID_DIGITS:
        return False

    if len(word) > MAX_PATIENT_ID_DIGITS:
        return False

    return True


def find_id_after_patient_word(words: list[str]) -> str | None:
    """
    Find a number that comes right after "patient", like "patient 1210".
    Returns None if there isn't one.
    """
    for index in range(len(words) - 1):
        word = words[index]
        next_word = words[index + 1]

        if word not in PATIENT_WORDS:
            continue

        if is_patient_id(word=next_word):
            return next_word

    return None


def find_id_that_is_not_a_dose(words: list[str]) -> str | None:
    """
    Find a 3 to 5 digit number that is not a dose.
    A number followed by "mg" (like "100 mg") is a dose, not a patient ID.
    """
    for index in range(len(words)):
        word = words[index]

        if not is_patient_id(word=word):
            continue

        is_last_word = index == len(words) - 1

        if is_last_word:
            return word

        next_word = words[index + 1]

        if next_word in DOSE_UNITS:
            continue

        return word

    return None


def find_patient_id(user_input: str) -> str | None:

    words = clean_words(user_input=user_input)

    patient_id = find_id_after_patient_word(words=words)

    if patient_id:
        return patient_id

    return find_id_that_is_not_a_dose(words=words)