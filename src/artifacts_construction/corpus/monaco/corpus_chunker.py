import json
import re
import copy
import argparse
from typing import List, Dict, Any
import tiktoken
from tqdm import tqdm


_ENCODER_CACHE = {}


def get_encoder(name: str = "cl100k_base"):
    
    """
    Return a cached tiktoken encoder, creating it on first use.
    """

    enc = _ENCODER_CACHE.get(name)
    if enc is None:
        enc = tiktoken.get_encoding(name)
        _ENCODER_CACHE[name] = enc
    return enc


def count_tokens(text: str, method: str) -> int:

    """
    Count the length of text in characters or tiktoken tokens.
    """

    if method == "char":
        return len(text)
    elif method == "tiktoken":
        return len(get_encoder().encode(text))
    else:
        raise ValueError(f"Unknown token method: {method}")


def find_sentence_boundaries(text: str) -> List[int]:
    """Return a list of end positions after sentence-ending punctuation."""
    return [m.end() for m in re.finditer(r'[.!?。？！]\s*', text)]


def chunk_sentence(text: str, max_length: int, token_method: str = "char") -> List[str]:
    
    """
    Split prose text into chunks that each fit within max_length.

    Preserves any leading Markdown headers in every chunk. Splits at
    sentence boundaries where possible; falls back to fixed-size slices
    if no boundaries are found.
    """

    if count_tokens(text, token_method) <= max_length:
        return [text]

    header_match = re.match(r'^((?:#[^\n]*\n)+\n?)', text)
    header = header_match.group(1) if header_match else ""
    content = text[len(header):]

    boundaries = find_sentence_boundaries(content)

    if not boundaries:
        return [text[i:i + max_length] for i in range(0, len(text), max_length)]

    chunks = []
    current_start = 0

    while current_start < len(content):
        chunk_text = header + content[current_start:]

        if count_tokens(chunk_text, token_method) <= max_length:
            chunks.append(chunk_text)
            break

        best_end = current_start
        for boundary in boundaries:
            if boundary <= current_start:
                continue
            if count_tokens(header + content[current_start:boundary], token_method) <= max_length:
                best_end = boundary
            else:
                break

        if best_end == current_start:
            for boundary in boundaries:
                if boundary > current_start:
                    best_end = boundary
                    break
            if best_end == current_start:
                best_end = len(content)

        chunks.append((header + content[current_start:best_end]).strip())
        current_start = best_end

    return [c for c in chunks if c.strip()]


def chunk_by_line(text: str, max_length: int, token_method: str = "char") -> List[str]:
    
    """
    Split line-based content (lists, infoboxes) into chunks that each fit within max_length.

    Preserves any leading Markdown headers in every chunk.
    """

    if count_tokens(text, token_method) <= max_length:
        return [text]

    header_match = re.match(r'^((?:#[^\n]*\n)+\n?)', text)
    header = header_match.group(1) if header_match else ""
    content = text[len(header):]

    lines = content.split('\n')
    chunks = []
    current_lines = []
    current_length = count_tokens(header, token_method)

    for i, line in enumerate(lines):
        line_with_newline = line + '\n' if i < len(lines) - 1 else line
        line_length = count_tokens(line_with_newline, token_method)

        if current_length + line_length <= max_length:
            current_lines.append(line_with_newline)
            current_length += line_length
        else:
            if current_lines:
                chunks.append(header + ''.join(current_lines))
            current_lines = [line_with_newline]
            current_length = count_tokens(header, token_method) + line_length

    if current_lines:
        chunks.append(header + ''.join(current_lines))

    return [c.rstrip('\n') for c in chunks if c.strip()]


def chunk_table(text: str, max_length: int, token_method: str = "char") -> List[str]:
    
    """
    Split a Markdown table into chunks that each fit within max_length.

    Every chunk repeats the full table header (Markdown headings + column
    headers + separator row) so each chunk is self-contained.
    """

    if count_tokens(text, token_method) <= max_length:
        return [text]

    header_match = re.match(r'^((?:#[^\n]*\n)+\n?)', text)
    md_header = header_match.group(1) if header_match else ""
    content = text[len(md_header):]

    table_header_lines = []
    data_lines = []
    found_table = False
    found_separator = False

    for line in content.split('\n'):
        stripped = line.strip()
        if stripped.startswith('|'):
            found_table = True
            if not found_separator:
                table_header_lines.append(line)
                if re.match(r'^[\|\-:\s]+$', stripped):
                    found_separator = True
            else:
                data_lines.append(line)
        else:
            if not found_table:
                md_header += line + '\n'
            else:
                data_lines.append(line)

    full_header = md_header + '\n'.join(table_header_lines)

    if not data_lines:
        return [text]

    header_length = count_tokens(full_header + '\n', token_method)
    chunks = []
    current_lines = []
    current_length = header_length

    for line in data_lines:
        line_with_newline = '\n' + line
        line_length = count_tokens(line_with_newline, token_method)

        if current_length + line_length <= max_length:
            current_lines.append(line)
            current_length += line_length
        else:
            if current_lines:
                chunks.append(full_header + '\n' + '\n'.join(current_lines))
            current_lines = [line]
            current_length = header_length + count_tokens(line_with_newline, token_method)

    if current_lines:
        chunks.append(full_header + '\n' + '\n'.join(current_lines))

    return [c for c in chunks if c.strip()]


def process_item(item: Dict[str, Any], max_length: int, token_method: str = "char") -> List[Dict[str, Any]]:
   
    """
    Split a single corpus item into sub-chunks if its text exceeds max_length.

    Chooses the chunking strategy based on content_type. When an item is
    split, each sub-chunk gets an updated document_id and chunk_index of
    the form '{original}.{i}', and the original index is preserved under
    original_chunk_index.
    """

    text = item.get("text", "")
    content_type = item.get("content_type", "sentence")

    if content_type == "sentence":
        chunks = chunk_sentence(text, max_length, token_method)
    elif content_type in ("list", "infobox"):
        chunks = chunk_by_line(text, max_length, token_method)
    elif content_type == "table":
        chunks = chunk_table(text, max_length, token_method)
    else:
        chunks = chunk_sentence(text, max_length, token_method)

    original_chunk_index = item.get("chunk_index", 0)
    result = []

    for i, chunk_text in enumerate(chunks):
        new_item = copy.deepcopy(item)
        new_item["text"] = chunk_text

        if len(chunks) > 1:
            new_item["chunk_index"] = f"{original_chunk_index}.{i}"
            new_item["original_chunk_index"] = original_chunk_index
            new_item["sub_chunk_index"] = i
            new_item["total_sub_chunks"] = len(chunks)
            new_item["document_id"] = f"{item.get('document_id', '')}.{i}"

        result.append(new_item)

    return result


def stream_process_jsonl(input_file: str, output_file: str, max_length: int, token_method: str = "char") -> None:
    
    """
    Read corpus items from input_file, chunk each one, and write results to output_file.
    """

    in_count = 0
    out_count = 0

    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc="Chunking"):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            in_count += 1
            for out_item in process_item(item, max_length, token_method):
                fout.write(json.dumps(out_item, ensure_ascii=False) + "\n")
                out_count += 1

    print(f"Done: {in_count:,} items -> {out_count:,} chunks")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk a corpus JSONL file to a maximum token length.")
    parser.add_argument("--input", required=True, help="Path to input .jsonl file")
    parser.add_argument("--output", required=True, help="Path to output .jsonl file")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum chunk length (default: 512)")
    parser.add_argument("--token-method", choices=["char", "tiktoken"], default="tiktoken", help="How to measure length: 'char' for characters, 'tiktoken' for tokens (default: tiktoken)")
    args = parser.parse_args()

    stream_process_jsonl(args.input, args.output, args.max_length, args.token_method)
