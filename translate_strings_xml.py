import re
import sys
from pathlib import Path
from deep_translator import GoogleTranslator
from tqdm import tqdm

SOURCE_LANG = "en"
TARGET_LANG = "es"

INNER_TEXT_RE = re.compile(r">(.*?)<", re.DOTALL)

# Conserva placeholders/escapes
PLACEHOLDER_RE = re.compile(r"(%\d+\$[sdif]|%[sdif]|\\n|\\t|\\r)")

SEP = "\n<<<SEG>>>\n"         # separador para dividir traducciones
MAX_CHARS = 3500             # tamaño por request (seguro)

def protect_tokens(text):
    token_map = {}
    idx = 0
    def repl(m):
        nonlocal idx
        key = f"__TOK{idx}__"
        token_map[key] = m.group(0)
        idx += 1
        return key
    return PLACEHOLDER_RE.sub(repl, text), token_map

def unprotect_tokens(text, token_map):
    for k, v in token_map.items():
        text = text.replace(k, v)
    return text

def decode_auto(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16")
    return raw.decode("utf-8")

def main():
    if len(sys.argv) < 3:
        print("Uso: python translate_strings_xml.py input.xml output.xml")
        sys.exit(1)

    in_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    content = decode_auto(in_path)

    matches = list(INNER_TEXT_RE.finditer(content))
    inners = [m.group(1) for m in matches]

    translator = GoogleTranslator(source=SOURCE_LANG, target=TARGET_LANG)

    # Protege placeholders
    protected_list = []
    token_maps = []
    for s in inners:
        p, mp = protect_tokens(s)
        protected_list.append(p)
        token_maps.append(mp)

    # Traducir en batches
    translated_list = []
    batch = []
    batch_len = 0
    idxs = []

    def flush_batch():
        nonlocal batch, batch_len, idxs, translated_list
        if not batch:
            return
        payload = SEP.join(batch)
        out = translator.translate(payload)
        parts = out.split(SEP)

        # Si el traductor rompió el separador, fallback a 1x1 para este batch
        if len(parts) != len(batch):
            parts = [translator.translate(x) for x in batch]

        translated_list.extend(parts)
        batch = []
        batch_len = 0
        idxs = []

    for s in tqdm(protected_list, desc="Batching", unit="texto"):
        s_len = len(s)
        # si el texto solo son espacios, no traduzcas
        if not s.strip():
            translated_list.append(s)
            continue

        # si se pasa, vaciamos batch
        if batch and (batch_len + len(SEP) + s_len) > MAX_CHARS:
            flush_batch()

        # si un solo texto es enorme, tradúcelo solo
        if s_len > MAX_CHARS:
            translated_list.append(translator.translate(s))
            continue

        batch.append(s)
        batch_len += (s_len + (len(SEP) if batch_len else 0))

    flush_batch()

    # Restaurar tokens
    translated_list = [
        unprotect_tokens(t, mp) for t, mp in zip(translated_list, token_maps)
    ]

    # Reconstruir archivo
    result = []
    last_end = 0
    for m, trans in zip(matches, translated_list):
        result.append(content[last_end:m.start()])
        result.append(f">{trans}<")
        last_end = m.end()
    result.append(content[last_end:])

    out_path.write_text("".join(result), encoding="utf-8", newline="\n")
    print(f"\n✔ Listo: {out_path}")

if __name__ == "__main__":
    main()
