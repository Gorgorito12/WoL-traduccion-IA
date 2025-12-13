# WoL-traduccion-IA

Pequeño script en Python para traducir archivos `strings.xml` entre idiomas (por defecto, de inglés a español) conservando los placeholders de formato.

## Uso

```bash
python translate_strings_xml.py entrada.xml salida.xml --source en --target es
```

### Opciones

- `--source`: idioma de origen (por defecto `en`).
- `--target`: idioma de destino (por defecto `es`).
- `--max-chars`: límite de caracteres por solicitud al traductor (por defecto 3500).

El script detecta automáticamente si el archivo de entrada está en UTF-8 o UTF-16 y escribe la salida en UTF-8 con saltos de línea Unix.
