"""Script para traducir archivos strings.xml conservando placeholders."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

from deep_translator import GoogleTranslator
from tqdm import tqdm

DEFAULT_SOURCE_LANG = "en"
DEFAULT_TARGET_LANG = "es"

INNER_TEXT_RE = re.compile(r">(.*?)<", re.DOTALL)

# Conserva placeholders/escapes
PLACEHOLDER_RE = re.compile(r"(%\d+\$[sdif]|%[sdif]|\\n|\\t|\\r)")

SEP = "\n<<<SEG>>>\n"  # separador para dividir traducciones
MAX_CHARS = 3500  # tamaño por request (seguro)


def protect_tokens(text: str) -> Tuple[str, Dict[str, str]]:
    """Reemplaza placeholders por tokens temporales.

    Devuelve el texto protegido y un mapa token->valor original para restaurar.
    """

    token_map: Dict[str, str] = {}
    idx = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal idx
        key = f"__TOK{idx}__"
        token_map[key] = match.group(0)
        idx += 1
        return key

    return PLACEHOLDER_RE.sub(repl, text), token_map


def unprotect_tokens(text: str, token_map: Dict[str, str]) -> str:
    for key, value in token_map.items():
        text = text.replace(key, value)
    return text


def decode_auto(path: Path) -> str:
    """Lee respetando BOM UTF-16; cae en UTF-8 si no existe."""

    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16")
    return raw.decode("utf-8")


def build_translator(source_lang: str, target_lang: str) -> GoogleTranslator:
    return GoogleTranslator(source=source_lang, target=target_lang)


def yield_batches(strings: Iterable[str], max_chars: int) -> Iterator[List[str]]:
    """Genera lotes que respetan el límite de caracteres del traductor."""

    batch: List[str] = []
    length = 0

    for text in strings:
        text_len = len(text)

        if batch and (length + len(SEP) + text_len) > max_chars:
            yield batch
            batch = []
            length = 0

        if text_len > max_chars:
            yield [text]
            batch = []
            length = 0
            continue

        batch.append(text)
        length += text_len + (len(SEP) if length else 0)

    if batch:
        yield batch


def translate_batch(translator: GoogleTranslator, batch: List[str]) -> List[str]:
    payload = SEP.join(batch)
    raw_output = translator.translate(payload)
    parts = raw_output.split(SEP)

    if len(parts) != len(batch):
        return [translator.translate(item) for item in batch]
    return parts


def translate_strings(
    inners: Iterable[str],
    source_lang: str = DEFAULT_SOURCE_LANG,
    target_lang: str = DEFAULT_TARGET_LANG,
    max_chars: int = MAX_CHARS,
) -> List[str]:
    translator = build_translator(source_lang, target_lang)

    protected: List[str] = []
    token_maps: List[Dict[str, str]] = []
    for inner in inners:
        protected_text, token_map = protect_tokens(inner)
        protected.append(protected_text)
        token_maps.append(token_map)

    translated: List[str] = []
    for batch in tqdm(yield_batches(protected, max_chars), desc="Traduciendo", unit="lote"):
        empty_mask = [not item.strip() for item in batch]
        batch_translation = translate_batch(translator, batch)

        for original, translated_item, is_empty in zip(batch, batch_translation, empty_mask):
            translated.append(original if is_empty else translated_item)

    return [unprotect_tokens(text, mp) for text, mp in zip(translated, token_maps)]


def rebuild_content(content: str, matches: List[re.Match[str]], translated: List[str]) -> str:
    result: List[str] = []
    last_end = 0

    for match, trans in zip(matches, translated):
        result.append(content[last_end:match.start()])
        result.append(f">{trans}<")
        last_end = match.end()
    result.append(content[last_end:])
    return "".join(result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Archivo strings.xml de entrada")
    parser.add_argument("output", type=Path, help="Ruta del archivo traducido")
    parser.add_argument("--source", default=DEFAULT_SOURCE_LANG, help="Idioma origen (por defecto: en)")
    parser.add_argument("--target", default=DEFAULT_TARGET_LANG, help="Idioma destino (por defecto: es)")
    parser.add_argument("--max-chars", type=int, default=MAX_CHARS, help="Máximo de caracteres por request")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise SystemExit(f"No se encontró el archivo de entrada: {args.input}")

    content = decode_auto(args.input)
    matches = list(INNER_TEXT_RE.finditer(content))
    inners = [match.group(1) for match in matches]

    translated = translate_strings(
        inners,
        source_lang=args.source,
        target_lang=args.target,
        max_chars=args.max_chars,
    )

    output_content = rebuild_content(content, matches, translated)
    args.output.write_text(output_content, encoding="utf-8", newline="\n")
    print(f"\n✔ Listo: {args.output}")


if __name__ == "__main__":
    main()
