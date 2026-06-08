from .markup import *
from typing import Optional, List


def has_header_row(table) -> bool:

    """
    Return True if the table's first content row uses header cells (!).
    """
    
    skip = ('{|', '|}', '|-', '|+')
    for line in table.string.split('\n'):
        line = line.strip()
        if not line or line.startswith(skip):
            continue
        return line.startswith('!')
    return False


def get_max_header_rowspan(table) -> int:
    
    """
    Return the maximum rowspan value found in the first header row.
    """
    
    try:
        table_str = str(table)
        lines = table_str.split('\n')
        
        max_rowspan = 1
        in_header_section = False
        found_first_header_row = False
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('|-'):
                in_header_section = True
                continue
            
            if in_header_section and line.startswith('!'):
                if found_first_header_row:
                    break
                
                found_first_header_row = True
                
                matches = re.findall(r'rowspan\s*=\s*["\']?(\d+)', line, re.IGNORECASE)
                for match in matches:
                    rowspan = int(match)
                    max_rowspan = max(max_rowspan, rowspan)
            
            elif in_header_section and line.startswith('|') and not line.startswith('|-'):
                break
        
        return max_rowspan
    except Exception:
        return 1


def is_style_only_text(s: str) -> bool:

    """
    Return True if s consists entirely of CSS declarations with no semantic text.
    """

    if not s:
        return True

    s = s.strip()

    decls = [d.strip() for d in s.split(';') if d.strip()]
    if not decls:
        return True

    allowed_keys = {
        "background", "background-color", "color",
        "border", "border-color", "border-width", "border-style",
        "width", "height", "min-width", "max-width", "min-height", "max-height",
        "text-align", "vertical-align", "align", "display"
    }

    for d in decls:
        if ':' not in d:
            return False
        key, val = d.split(':', 1)
        key = key.strip().lower()
        val = val.strip()

        if key not in allowed_keys:
            return False

    return True


def parse_bartable(cell_str: str, width: int) -> List[dict]:

    """
    Expand a {{bartable}} cell into a list of width cell dicts.
    """

    if width <= 0:
        return []

    s = (cell_str or "").strip()

    if not re.search(r"\{\{\s*bartable", s, re.IGNORECASE):
        first = {"text": clean_wiki_markup(s)}
        return [first] + [{"text": ""} for _ in range(width - 1)]

    m = re.search(r"\{\{\s*bartable\s*(.*?)\}\}", s, re.IGNORECASE | re.DOTALL)
    if not m:
        return [{"text": ""} for _ in range(width)]

    inside = m.group(1)
    raw_parts = [p.strip() for p in inside.split("|")]

    while raw_parts and raw_parts[0] == "":
        raw_parts.pop(0)

    val = raw_parts[0] if len(raw_parts) >= 1 else ""
    unit = raw_parts[1] if len(raw_parts) >= 2 else ""

    value_text = clean_wiki_markup(((val + unit).strip() if (val or unit) else ""))

    style = ""
    for p in reversed(raw_parts):
        p = p.strip()
        if not p:
            continue
        if ("background" in p.lower()) or ("color" in p.lower()) or (":" in p):
            style = p
            break

    style_text = style.strip()
    if is_style_only_text(style_text):
        style_text = ""

    out = [{"text": ""} for _ in range(width)]
    out[0] = {"text": value_text}

    if width >= 2:
        out[1] = {"text": style_text}

    return out


def cell_to_lines_from_text(text: str) -> List[str]:

    """
    Split a cell's text into lines, normalizing whitespace and dropping blank lines.
    """

    if text is None:
        return [""]

    text = re.sub(r'(&nbsp;|&#160;|\u00A0)+', ' ', text)
    lines = [ln.strip() for ln in text.split('\n')]
    lines = [ln for ln in lines if ln != ""]
    return lines if lines else [""]


def md_escape_cell(s: str) -> str:

    """
    Escape pipe characters and replace newlines for safe use inside a Markdown table cell.
    """

    s = s or ""
    s = s.replace("|", r"\|")
    s = s.replace("\n", "<br>")
    return s.strip()


def is_fake_col_header(h: str) -> bool:
    
    """
    Return True if h is an auto-generated placeholder header like col1, col2.
    """
    
    return bool(re.match(r"^col\d+$", (h or "").strip(), re.IGNORECASE))


def make_md_row(cells) -> str:

    """
    Render a list of cell strings as a Markdown table row.
    """

    return "| " + " | ".join(md_escape_cell(v) for v in cells) + " |"
 

def table_to_markdown(table) -> Optional[str]:

    """
    Convert a wikitextparser Table object to a Markdown table string.
    """

    try:
        data_span = table.data(span=True)
        data_raw  = table.data(span=False)
 
        if not data_span:
            return None
 
        has_header = has_header_row(table)
        has_bartable = any(
            re.search(r"\{\{\s*bartable", str(cell or ""), re.IGNORECASE)
            for row in data_span for cell in (row or [])
        )
 
        # --- no-header table ---
        if not has_header or len(data_span) < 2:
            data = data_raw or data_span
            text_rows = [
                [clean_wiki_markup(str(cell) if cell else '') for cell in row]
                for row in data
                if any(cell for cell in row)
            ]
            if not text_rows:
                return None
            ncols = max(len(r) for r in text_rows)
            lines = [make_md_row([""] * ncols)]
            lines.append("| " + " | ".join(["---"] * ncols) + " |")
            for row in text_rows:
                row += [""] * (ncols - len(row))
                lines.append(make_md_row(row))
            return "\n".join(lines)
 
        # --- header table ---
        header_rows_count = get_max_header_rowspan(table)
 
        if len(data_span) < header_rows_count + 1:
            return None
 
        # build flat headers + groups from first header row
        headers_expanded = [
            clean_wiki_markup(str(h) if h else '') or f"col{i+1}"
            for i, h in enumerate(data_span[0])
        ]
 
        groups = []
        for h in headers_expanded:
            if groups and groups[-1][0] == h:
                groups[-1] = (groups[-1][0], groups[-1][1] + 1)
            else:
                groups.append((h, 1))
 
        flat_headers = []
        for name, width in groups:
            if width == 1:
                flat_headers.append(name)
            elif has_bartable and width == 2:
                flat_headers.extend([name, f"{name}_bar"])
            else:
                flat_headers.extend([f"{name}_{j+1}" for j in range(width)])
 
        ncols = len(flat_headers)
 
        # md headers (drop fake col* placeholders)
        md_headers = [
            "" if is_fake_col_header(h) else h
            for h in flat_headers
        ]
 
        # body rows
        body_src = data_raw if has_bartable else data_span
        body_rows = body_src[header_rows_count:]
 
        def process_body_row(row_data) -> List[List[str]]:
            """Return one or more text rows (multiline cells expand to multiple rows)."""
            row_cells = []
            cell_idx = 0
 
            for hname, width in groups:
                if cell_idx >= len(row_data):
                    row_cells.extend([{"text": ""}] * width)
                    continue
 
                cell = row_data[cell_idx]
                cell_str = str(cell) if cell else ""
                cell_clean = clean_wiki_markup(cell_str, preserve_newlines=True)
 
                if width == 1:
                    row_cells.append({"text": cell_clean})
                    cell_idx += 1
                elif has_bartable and re.search(r"\{\{\s*bartable", cell_str, re.IGNORECASE):
                    row_cells.extend(parse_bartable(cell_str, width))
                    cell_idx += 1
                else:
                    for j in range(width):
                        if cell_idx + j < len(row_data):
                            c = row_data[cell_idx + j]
                            c_clean = clean_wiki_markup(str(c) if c else "", preserve_newlines=True)
                            row_cells.append({"text": c_clean})
                        else:
                            row_cells.append({"text": ""})
                    cell_idx += width
 
            # pad / trim to ncols
            row_cells = (row_cells + [{"text": ""}] * ncols)[:ncols]
 
            # expand multiline cells
            cells_lines = [cell_to_lines_from_text(c.get("text", "")) for c in row_cells]
            max_lines = max(len(ls) for ls in cells_lines)
 
            result = []
            for line_i in range(max_lines):
                expanded = [ls[line_i] if line_i < len(ls) else "" for ls in cells_lines]
                if any(expanded):
                    result.append(expanded)
            return result
 
        all_body_text_rows = []
        for row_data in body_rows:
            if row_data:
                all_body_text_rows.extend(process_body_row(row_data))
 
        # drop entirely-empty columns (including _bar columns)
        keep_cols = [
            j for j in range(ncols)
            if not (
                all((r[j] or "").strip() == "" for r in all_body_text_rows)
            )
        ]
        if not keep_cols:
            return None
 
        md_headers   = [md_headers[j] for j in keep_cols]
        all_body_text_rows = [[r[j] for j in keep_cols] for r in all_body_text_rows]
 
        # assemble markdown
        lines = [make_md_row(md_headers)]
 
        # extra header rows (rowspan > 1)
        extra_header_text_rows = []
        for row_idx in range(1, header_rows_count):
            if row_idx >= len(data_span):
                break
            row = [
                clean_wiki_markup(str(cell) if cell else "")
                for cell in data_span[row_idx]
            ]
            row = (row + [""] * ncols)[:ncols]
            extra_header_text_rows.append([row[j] for j in keep_cols])
 
        for row in extra_header_text_rows:
            lines.append(make_md_row(row))
 
        lines.append("| " + " | ".join(["---"] * len(md_headers)) + " |")
 
        for row in all_body_text_rows:
            lines.append(make_md_row(row))
 
        return "\n".join(lines)
 
    except Exception:
        return None

