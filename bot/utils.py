def remove_description_table(text: str) -> str:
    index = text.find("DESCRIPTION TABLE")

    if index != -1:
        return text[:index]

    return text
