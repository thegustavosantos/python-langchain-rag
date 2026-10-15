def recursive_text_splitter(text, max_length=600, overlap=150):
    if len(text) <= max_length:
        return [text]

    # Tentamos encontrar um ponto de separação preferencial
    break_point = text.rfind(' ', 0, max_length)
    if break_point == -1:
        break_point = max_length

    # Definindo o chunk atual
    chunk = text[:break_point]
    # Definindo o restante do texto, aplicando overlap
    remainder = text[max(0, break_point - overlap):]

    return [chunk] + recursive_text_splitter(remainder, max_length, overlap)

# Exemplo de uso
documento = "Texto longo que precisa ser dividido em vários pedaços de forma inteligente..."
chunks = recursive_text_splitter(documento)
for i, c in enumerate(chunks):
    print(f"Chunk {i+1}: {c}\n")
