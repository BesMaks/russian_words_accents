from typing import List

# Define sounds by types
vowel_sounds = "А, О, У, Ы, Э, Е, Ё, И, Ю, Я".lower().split(", ")
consonant_sounds = "Б, В, Г, Д, Ж, З, Й, К, Л, М, Н, П, Р, С, Т, Ф, Х, Ц, Ч, Ш, Щ".lower().split(", ")
sonorous_sounds = "р, л, м, н, й".split(", ")

def split(word: str) -> List[str]:
    word_length = len(word)
    syllables_list = []
    cur_syllable = ""
    skip = False

    for i, letter in enumerate(word):
        if skip:
            skip = False
            continue

        cur_syllable += letter

        # If it's the last letter in the word
        if i == word_length - 1:
            if letter in consonant_sounds or letter in ["ь", "ъ"]:
                syllables_list[-1] += cur_syllable
            else:
                syllables_list.append(cur_syllable)
            return syllables_list

        # If the letter is 'ь' or 'ъ'
        if letter in ("ь", "ъ"):
            if len(cur_syllable) == 1:
                syllables_list[-1] += letter
                cur_syllable = ""
            continue

        # If the letter is a consonant
        if letter in consonant_sounds:
            continue

        # If the next letter is a vowel or not a sonorous sound,
        # or the letter after next is a vowel
        if (i + 1 < word_length and (word[i + 1] in vowel_sounds or word[i + 1] not in sonorous_sounds)
                or i + 2 < word_length and word[i + 2] in vowel_sounds):
            syllables_list.append(cur_syllable)
            cur_syllable = ""
            continue

        # If the next letter is a consonant or a sonorous sound
        cur_syllable += word[i + 1]
        syllables_list.append(cur_syllable)
        cur_syllable = ""
        skip = True

    return syllables_list

if __name__ == "__main__":
    print(split("дифференциальный"))
