import re
import wikitextparser as wtp
from typing import Optional, List, Dict, Any
from .markup import clean_wiki_markup
from .table import table_to_markdown


SKIP_SECTIONS = {
    'see also', 'references', 'external links',
    'further reading', 'notes', 'citations',
    'bibliography', 'sources'
}


def split_sections(wikitext: str) -> List[Dict[str, Any]]:
    
    """
    Split wikitext into a flat list of sections, each with its own content only.
    """

    sections = []

    heading_pattern = re.compile(r'^(={2,6})\s*(.+?)\s*\1\s*$', re.MULTILINE)
    headings = []
    for match in heading_pattern.finditer(wikitext):
        level = len(match.group(1))
        title = match.group(2).strip()
        headings.append({
            'level': level,
            'title': title,
            'heading_start': match.start(),
            'content_start': match.end()
        })

    lead_content = wikitext[:headings[0]['heading_start']].strip() if headings else wikitext.strip()
    if lead_content:
        sections.append({'title': '', 'level': 1, 'content': lead_content})

    for i, heading in enumerate(headings):
        content_end = headings[i + 1]['heading_start'] if i + 1 < len(headings) else len(wikitext)
        content = wikitext[heading['content_start']:content_end].strip()
        sections.append({
            'title': heading['title'],
            'level': heading['level'],
            'content': content
        })

    return sections


def is_wrapper_table(table) -> bool:

    """
    Return True if the table is just a container for nested tables and has no direct content.
    """

    try:
        data = table.data(span=False)
        if not data:
            return False
        
        for row in data:
            for cell in row:
                cell_str = str(cell) if cell else ''
                if '{|' in cell_str:
                    return True
        
        return False
    except:
        return False


def extract_infobox_text(parsed) -> Optional[str]:

    """
    Extract infobox fields as plain 'key: value' lines.
    """

    try:
        skip_fields = {
            'image', 'image_size', 'imagesize', 'image_caption', 'caption',
            'imageclass', 'image_upright', 'image_alt', 'image2', 'logo',
            'map', 'mapsize', 'map_caption', 'pushpin_map',
            'hide header', 'header caption',
            'child', 'subbox', 'decat', 'embedded',
        }
        skip_prefixes = (
            'fam', 'image', 'map', 'logo', 'flag', 'seal', 'symbol',
            'section',
        )
        
        lines = []
        seen_keys = set()
        
        for template in parsed.templates:
            tname = template.name.strip().lower()
            if 'infobox' not in tname:
                continue
            
            for arg in template.arguments:
                key = arg.name.strip() if arg.name else ''
                key_lower = key.lower()
                
                if key_lower in skip_fields:
                    continue
                
                if any(key_lower.startswith(prefix) and 
                       (len(key_lower) == len(prefix) or key_lower[len(prefix):].isdigit()) 
                       for prefix in skip_prefixes):
                    continue
                
                val = arg.value.strip() if arg.value else ''
                val = clean_wiki_markup(val)
                
                if not val or val.startswith('|') or val.startswith(']') or 'class=' in val.lower():
                    continue
                
                if key and val:
                    display_key = key
                    for prefix in ['Ship ', 'ship_']:
                        if key.startswith(prefix):
                            display_key = key[len(prefix):]
                            break
                    
                    if display_key not in seen_keys:
                        seen_keys.add(display_key)
                        readable_key = display_key.replace('_', ' ')
                        lines.append(f"{readable_key}: {val}")
        
        if lines:
            return '\n'.join(lines)
        return None
    except:
        return None


def extract_lists_from_content(content: str) -> List[str]:

    """
    Extract all lists from a section and return them as '* item' formatted strings.
    """

    try:
        parsed = wtp.parse(content)
        result = []
        
        for lst in parsed.get_lists():
            items = []
            for item in lst.items:
                item_clean = clean_wiki_markup(str(item))
                if item_clean:
                    items.append(f"* {item_clean}")
            
            if items:
                result.append('\n'.join(items))
        
        return result
    except:
        return []


def extract_plain_text(content: str) -> str:

    """
    Extract prose text from a section, stripping out tables and lists.
    """

    try:
        parsed = wtp.parse(content)
        
        for table in parsed.tables:
            content = content.replace(table.string, '')
        
        for lst in parsed.get_lists():
            content = content.replace(lst.string, '')
        
        parsed = wtp.parse(content)
        text = parsed.plain_text(
            replace_templates=True,
            replace_wikilinks=True,
            replace_external_links=True,
            replace_tags=True,
            replace_bolds_and_italics=True
        )
        
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except:
        return ''


def extract_tables_from_content(content: str):

    """
    Extract all tables from a section as Markdown strings, skipping wrapper tables.
    """

    try:
        parsed = wtp.parse(content)
        result = []
        
        for table in parsed.tables:
            if is_wrapper_table(table):
                continue
            
            table_data = table_to_markdown(table)
            if table_data:
                result.append(table_data)
        
        return result
    except:
        return []


def build_section_header(title: str, level_titles: Dict[int, str], current_level: int) -> str:
    
    """
    Build a Markdown header block showing the full section hierarchy.
    """

    if not title:
        return ""
    
    header_parts = [f"# {title}"]
    
    for level in sorted([k for k in level_titles.keys() if k <= current_level]):
        section_title = level_titles[level]
        header_parts.append(f"{'#' * level} {section_title}")
    
    return '\n'.join(header_parts) + '\n\n'
