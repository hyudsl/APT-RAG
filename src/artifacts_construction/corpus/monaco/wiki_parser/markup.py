import re


SUPERSCRIPT_MAP = {
    '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
    '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
    '+': '⁺', '-': '⁻', '=': '⁼', '(': '⁽', ')': '⁾',
    'a': 'ᵃ', 'b': 'ᵇ', 'c': 'ᶜ', 'd': 'ᵈ', 'e': 'ᵉ',
    'f': 'ᶠ', 'g': 'ᵍ', 'h': 'ʰ', 'i': 'ⁱ', 'j': 'ʲ',
    'k': 'ᵏ', 'l': 'ˡ', 'm': 'ᵐ', 'n': 'ⁿ', 'o': 'ᵒ',
    'p': 'ᵖ', 'r': 'ʳ', 's': 'ˢ', 't': 'ᵗ', 'u': 'ᵘ',
    'v': 'ᵛ', 'w': 'ʷ', 'x': 'ˣ', 'y': 'ʸ', 'z': 'ᶻ',
}

SUBSCRIPT_MAP = {
    '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄',
    '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉',
    '+': '₊', '-': '₋', '=': '₌', '(': '₍', ')': '₎',
    'a': 'ₐ', 'e': 'ₑ', 'h': 'ₕ', 'i': 'ᵢ', 'j': 'ⱼ',
    'k': 'ₖ', 'l': 'ₗ', 'm': 'ₘ', 'n': 'ₙ', 'o': 'ₒ',
    'p': 'ₚ', 'r': 'ᵣ', 's': 'ₛ', 't': 'ₜ', 'u': 'ᵤ',
    'v': 'ᵥ', 'x': 'ₓ',
}


def to_unicode_superscript(text: str) -> str:

    """
    Convert each character in text to its unicode superscript equivalent.
    """
    
    if not text:
        return ''
    result = []
    for char in text.lower():
        if char in SUPERSCRIPT_MAP:
            result.append(SUPERSCRIPT_MAP[char])
        elif char.isspace():
            result.append(char)
        else:
            result.append(char)
    return ''.join(result)


def to_unicode_subscript(text: str) -> str:

    """
    Convert each character in text to its unicode subscript equivalent.
    """

    if not text:
        return ''
    result = []
    for char in text.lower():
        if char in SUBSCRIPT_MAP:
            result.append(SUBSCRIPT_MAP[char])
        elif char.isspace():
            result.append(char)
        else:
            result.append(char)
    return ''.join(result)


def process_superscript(content: str) -> str:

    """
    Convert <sup> content to plain text.
    """

    content = content.strip()
    if not content:
        return ''
    
    if re.match(r'^\[[\w\s\-,]+\]$', content):
        return ''
    if any(keyword in content.lower() for keyword in ['citation', 'clarification', 'verification', 'note']):
        return ''
    if content.isdigit() and len(content) <= 4:
        return to_unicode_superscript(content)
    if content.lower() in ('st', 'nd', 'rd', 'th'):
        return to_unicode_superscript(content)
    if len(content) <= 3 and all(c in SUPERSCRIPT_MAP or c.isspace() for c in content.lower()):
        return to_unicode_superscript(content)
    if len(content) > 20:
        return ''
    
    return f"^{{{content}}}"


def process_subscript(content: str) -> str:

    """
    Convert <sub> content to plain text.
    """

    content = content.strip()
    if not content:
        return ''
    
    if len(content) <= 10 and all(c in SUBSCRIPT_MAP or c.isspace() for c in content.lower()):
        return to_unicode_subscript(content)
    if len(content) > 20:
        return ''
    
    return f"_{{{content}}}"


def clean_sub_sup_tags(text: str) -> str:

    """
    Replace <sup> and <sub> HTML tags with plain-text equivalents.
    """

    if not text:
        return text
    
    text = re.sub(
        r'<sup[^>]*>(.*?)</sup>',
        lambda m: process_superscript(m.group(1)),
        text,
        flags=re.IGNORECASE | re.DOTALL
    )
    text = re.sub(
        r'<sub[^>]*>(.*?)</sub>',
        lambda m: process_subscript(m.group(1)),
        text,
        flags=re.IGNORECASE | re.DOTALL
    )
    return text


def process_wikilink(match):

    """
    Convert a matched [[wikilink]] to its display text.
    """

    content = match.group(1)
    parts = content.split('|')
    if parts[0].strip().lower().startswith(('file:', 'image:')):
        for part in reversed(parts):
            part = part.strip()
            if part and not re.match(r'^x?\d+px$', part, re.IGNORECASE):
                if part.lower() not in ('thumb', 'thumbnail', 'frame', 'frameless', 'left', 'right', 'center', 'none', 'upright'):
                    if not part.lower().startswith(('file:', 'image:')):
                        return part
        return ''
    return parts[-1].strip() if parts else ''


def extract_template_text(match):
    
    """
    Convert a single matched {{template}} to plain text.
    """
    
    content = match.group(1)
    parts = content.split('|')
    template_name = parts[0].strip().lower() if parts else ''

    if template_name == 'percentage':
        if len(parts) >= 3:
            try:
                numerator = float(parts[1].strip().replace(',', ''))
                denominator = float(parts[2].strip().replace(',', ''))
                decimals = int(parts[3].strip()) if len(parts) >= 4 else 2
                
                percentage = (numerator / denominator) * 100
                return f"{percentage:.{decimals}f}%"
            except (ValueError, ZeroDivisionError):
                pass
        return '' 
    
    if 'formatnum' in template_name:
        if ':' in template_name:
            num_str = template_name.split(':', 1)[1].strip()
            return num_str
        elif len(parts) >= 2:
            return parts[1].strip()
        return ''
    
    if template_name == 'rnd':
        if len(parts) >= 3:
            try:
                number = float(parts[1].strip())
                decimals = int(parts[2].strip())
                return f"{number:.{decimals}f}"
            except (ValueError, IndexError):
                pass
        return ''
    
    if template_name in ('convert', 'coord', 'age', 'birth date', 'death date'):
        return ''
    
    extracted = []
    for i, part in enumerate(parts):
        part = part.strip()
        
        if i == 0:
            continue
        if '=' in part and not part.startswith('['):
            continue
        if part.startswith('File:') or part.startswith('Image:'):
            continue
        if not part:
            continue

        if re.match(r'^x?\d+px$', part, re.IGNORECASE):
            continue
        if re.match(r'^\d+%$', part):
            continue
        if part.lower() in ('thumb', 'thumbnail', 'frame', 'frameless', 'left', 'right', 'center', 'none', 'upright'):
            continue
        
        if i == 1 and re.match(r'^[a-z]{2,3}$', part):
            continue
        
        extracted.append(part)
    
    return ', '.join(extracted) if extracted else ''


def clean_wiki_markup(text: str, preserve_newlines: bool = False) -> str:
    
    """
    Strip all wiki markup from text and return plain text.

    - Removes references, templates, wikilinks, HTML tags, and table syntax.
    - By default collapses all whitespace to single spaces; pass
    - preserve_newlines=True to keep line breaks (useful for table cells).
    """ 
    
    if not text:
        return ''
    
    text = re.sub(r'\{\{formatnum:([^}]+)\}\}', r'\1', text, flags=re.IGNORECASE)
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'<ref[^>]*\s*/>', '', text, flags=re.IGNORECASE)

    if preserve_newlines:
        text = re.sub(r'\{\{(?:br|break|clear|-)(?:\|[^}]*)?\}\}', '\n', text, flags=re.IGNORECASE)
    else:
        text = re.sub(r'\{\{(?:br|break|clear|-)(?:\|[^}]*)?\}\}', ' ', text, flags=re.IGNORECASE)
    
    prev_text = None
    while prev_text != text:
        prev_text = text
        text = re.sub(r'\{\{([^{}]*)\}\}', extract_template_text, text)
    
    text = re.sub(r'\[\[([^\]]+)\]\]', process_wikilink, text)
    text = re.sub(r"'''?", '', text)
    text = re.sub(r'(?i)<\s*br\s*/?\s*>', '\n', text)
    text = re.sub(r'(&nbsp;|&#160;|\u00A0)+', ' ', text)

    text = clean_sub_sup_tags(text)
    text = re.sub(r'<[^>]+>', '', text)
    
    text = re.sub(r'File:[^\]|]+', '', text)
    text = re.sub(r'Image:[^\]|]+', '', text)
    
    text = re.sub(r'(?<![a-zA-Z])style\s*=\s*["\']?[^|,\n"\']*["\']?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[a-z]+style\s*=\s*["\']?[^|,\n"\']*["\']?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(?<![a-zA-Z])class\s*=\s*["\']?[^|,\n"\']+["\']?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(?<![a-zA-Z])bgcolor\s*=\s*["\']?[^|,\n"\']+["\']?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(?<![a-zA-Z])(?:align|valign|text-align)\s*=\s*["\']?[^|,\n"\']+["\']?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(?<![a-zA-Z])(?:width|height)\s*=\s*["\']?\d+%?["\']?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(?<![a-zA-Z])(?:colspan|rowspan)\s*=\s*["\']?\d+["\']?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(?<![a-zA-Z])scope\s*=\s*["\']?[^|,\n"\']+["\']?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(?<![a-zA-Z])data-[a-z-]+\s*=\s*["\']?[^|,\n"\']+["\']?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(?<![a-zA-Z])chset-(?:cell|ctrl)\d*', '', text, flags=re.IGNORECASE)
    
    text = re.sub(r'infobox\s*,\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(?<![a-zA-Z])(?:title|above|below|data\d*|label\d*|header\d*|image\d*)\s*=\s*', '', text, flags=re.IGNORECASE)

    text = re.sub(r'\{\|[^\n]*\n?', '', text)
    text = re.sub(r'\|\}', '', text)
    text = re.sub(r'\|\-[^\n]*\n?', '', text)
    
    text = re.sub(r'\s*!!\s*', ' | ', text)
    text = re.sub(r'\|\s*!\s*', '| ', text)
    
    text = re.sub(r'\|\s*#[0-9a-fA-F]{3,6}\s*,\s*', '| ', text)
    text = re.sub(r',\s*#[0-9a-fA-F]{3,6}\s*,', ',', text)
    text = re.sub(r'^\s*#[0-9a-fA-F]{3,6}\s*,\s*', '', text, flags=re.MULTILINE)
    
    text = re.sub(r'[\[\]{}]', '', text)
    
    text = re.sub(r'\|+', ', ', text)
    text = re.sub(r',\s*,+', ',', text)
    text = re.sub(r'^\s*,\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*,\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\|\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*\|\s*$', '', text, flags=re.MULTILINE)
    
    if preserve_newlines:
        text = re.sub(r'[ \t\r\f\v]+', ' ', text)
        text = '\n'.join(line.strip() for line in text.split('\n'))
        text = re.sub(r'\n{2,}', '\n', text)
    else:
        text = re.sub(r'\s+', ' ', text)
    return text.strip()
