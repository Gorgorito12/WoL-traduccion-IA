"""Script para traducir archivos strings.xml conservando placeholders."""

from __future__ import annotations

import argparse
import logging
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

from deep_translator import GoogleTranslator
from tqdm import tqdm

DEFAULT_SOURCE_LANG = "en"
DEFAULT_TARGET_LANG = "es"

# Conserva placeholders/escapes
PLACEHOLDER_RE = re.compile(r"(%\d+\$[sdif]|%[sdif]|\\n|\\t|\\r)")

SEP = "\n<<<SEG>>>\n"  # separador para dividir traducciones
MAX_CHARS = 3500  # tamaño por request (seguro)
DEFAULT_MAX_RETRIES = 3
BACKOFF_SECONDS = 2.0


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


def translate_batch(translator: GoogleTranslator, batch: Sequence[str]) -> List[str]:
    output = translator.translate_batch(batch)

    if not isinstance(output, list):
        raise ValueError(
            "El traductor no devolvió una lista de traducciones para el lote actual."
        )

    if len(output) != len(batch):
        raise ValueError(
            "El número de traducciones devuelto no coincide con el lote: "
            f"esperado {len(batch)}, recibido {len(output)}."
        )

    safe_output: List[str] = []
    for original, translated in zip(batch, output):
        translated_text = "" if translated is None else str(translated)
        safe_output.append(original if not translated_text.strip() else translated_text)

    return safe_output


def translate_batch_with_retry(
    translator: GoogleTranslator, batch: Sequence[str], max_retries: int
) -> List[str]:
    attempt = 0
    while True:
        try:
            return translate_batch(translator, batch)
        except Exception as exc:  # noqa: BLE001 - cualquier fallo debe reintentarse/controlarse
            attempt += 1
            if attempt > max_retries:
                raise RuntimeError(
                    "Fallaron los reintentos de traducción para el lote. "
                    f"Primero: {batch[0][:80]!r}, tamaño: {len(batch)}"
                ) from exc

            wait = BACKOFF_SECONDS * (2 ** (attempt - 1))
            logging.warning(
                "Error en traducción (intento %s/%s): %s. Reintentando en %.1fs...",
                attempt,
                max_retries,
                exc,
                wait,
            )
            time.sleep(wait)


def translate_strings(
    inners: Iterable[str],
    source_lang: str = DEFAULT_SOURCE_LANG,
    target_lang: str = DEFAULT_TARGET_LANG,
    max_chars: int = MAX_CHARS,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> List[str]:
    translator = build_translator(source_lang, target_lang)

    protected: List[str] = []
    token_maps: List[Dict[str, str]] = []
    for inner in inners:
        protected_text, token_map = protect_tokens(inner)
        protected.append(protected_text)
        token_maps.append(token_map)

    cache: Dict[str, str] = {}
    unique_to_translate: List[str] = []
    for text in protected:
        if not text.strip():
            cache[text] = text
            continue

        if text not in cache:
            cache[text] = ""
            unique_to_translate.append(text)

    batches = list(yield_batches(unique_to_translate, max_chars))

    for batch in tqdm(batches, desc="Traduciendo", unit="lote", total=len(batches)):
        batch_translation = translate_batch_with_retry(translator, batch, max_retries)

        if len(batch_translation) != len(batch):
            raise RuntimeError(
                "El número de traducciones devuelto no coincide con el lote tras reintentos."
            )

        for original, translated_item in zip(batch, batch_translation):
            cache[original] = translated_item

    translated = [cache[text] for text in protected]

    if len(translated) != len(token_maps):
        raise RuntimeError(
            "El número total de traducciones no coincide con los textos originales "
            f"({len(translated)} vs {len(token_maps)})."
        )

    return [unprotect_tokens(text, mp) for text, mp in zip(translated, token_maps)]


def parse_strings_xml(path: Path) -> ET.ElementTree:
    content = decode_auto(path)
    return ET.ElementTree(ET.fromstring(content))


def iter_translatable_elements(root: ET.Element) -> Iterator[ET.Element]:
    """Itera sobre nodos <string> y <plurals>/<item>, ignorando namespaces.

    Algunos archivos `strings.xml` incluyen namespaces XML; en esos casos
    ``ElementTree.findall(".//string")`` no encuentra coincidencias porque las
    etiquetas se representan como ``{namespace}string``. Aquí recorremos todo el
    árbol y comparamos por sufijo para que los namespaces no impidan traducir.
    """

    def tag_matches(tag: str, name: str) -> bool:
        return tag.split("}")[-1].lower() == name

    for elem in root.iter():
        if tag_matches(elem.tag, "string"):
            yield elem
        elif tag_matches(elem.tag, "plurals"):
            for item in elem:
                if tag_matches(item.tag, "item"):
                    yield item


def extract_texts(elements: Iterable[ET.Element]) -> List[str]:
    return [(elem.text or "") for elem in elements]


def update_elements_text(elements: Iterable[ET.Element], texts: Sequence[str]) -> None:
    for elem, text in zip(elements, texts):
        elem.text = text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Archivo strings.xml de entrada")
    parser.add_argument("output", type=Path, help="Ruta del archivo traducido")
    parser.add_argument("--source", default=DEFAULT_SOURCE_LANG, help="Idioma origen (por defecto: en)")
    parser.add_argument("--target", default=DEFAULT_TARGET_LANG, help="Idioma destino (por defecto: es)")
    parser.add_argument("--max-chars", type=int, default=MAX_CHARS, help="Máximo de caracteres por request")
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Número máximo de reintentos por lote (por defecto: 3)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    args = parse_args()

    if not args.input.exists():
        raise SystemExit(f"No se encontró el archivo de entrada: {args.input}")

    tree = parse_strings_xml(args.input)
    root = tree.getroot()
    elements = list(iter_translatable_elements(root))
    count = len(elements)
    print(f"Nodos traducibles encontrados: {count}")

    if count == 0:
        raise SystemExit("No se encontraron nodos traducibles en el XML; se aborta la traducción.")

    texts = extract_texts(elements)

    translated = translate_strings(
        texts,
        source_lang=args.source,
        target_lang=args.target,
        max_chars=args.max_chars,
        max_retries=args.max_retries,
    )

    if len(translated) != len(elements):
        raise RuntimeError(
            "No se pudo asignar las traducciones a los nodos del XML: "
            f"{len(translated)} traducciones para {len(elements)} nodos."
        )

    update_elements_text(elements, translated)
    tree.write(args.output, encoding="utf-8", xml_declaration=True)
    print(f"\n✔ Listo: {args.output}")


if __name__ == "__main__":
    main()
