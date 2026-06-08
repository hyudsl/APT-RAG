import bz2
import sys
import json
import time
import glob
import argparse
import wikitextparser as wtp
import xml.etree.ElementTree as ET
from urllib.parse import quote
from wiki_parser.extractor import *
from wiki_parser.markup import clean_wiki_markup



def should_process_page(page_id: str) -> bool:

    """
    Return True if this page_id falls within the current worker's assigned range.
    """

    try:
        pid = int(page_id)
        return WORKER_START <= pid < WORKER_END
    except:
        return False


def make_url(title: str) -> str:

    """
    Build a Wikipedia URL from a page title.
    """

    encoded = quote(str(title or '').replace(' ', '_'), safe='')
    return f"https://en.wikipedia.org/wiki/{encoded}"


def make_doc_id(page_id: str, title: str, section_title: str, content_type: str, chunk_index: int) -> str:
    
    """
    Build a unique document ID from page and section metadata.
    """

    return f"{page_id}:{title}:{section_title}:{content_type}:{chunk_index}"


def write_item(out_file, item, first_item: bool) -> bool:
    
    """
    Serialize item as a JSONL line and write it to out_file.
    """

    global ITEM_COUNT
    out_file.write(json.dumps(item, ensure_ascii=False) + '\n')
    ITEM_COUNT += 1


def process_page(page_id: str, title: str, ns: str, revid: str, text: str, out_file, first_item: bool) -> bool:

    """
    Extract and write all content items from a single Wikipedia page.
    """

    if not text or not title or str(ns) != '0':
        return
    if text.strip().upper().startswith('#REDIRECT'):
        return
    
    url = make_url(title)
    chunk_index = 0
    
    try:
        full_parsed = wtp.parse(text)
    except:
        return 
    
    try:
        infobox_data = extract_infobox_text(full_parsed)
        if infobox_data:
            write_item(out_file, {
                "document_id": make_doc_id(page_id, title, '', 'infobox', chunk_index),
                "page_id": str(page_id),
                "title": title,
                "url": url,
                "revid": str(revid),
                "language": 'en',
                "section": '',
                "section_level": 1,
                "content_type": 'infobox',
                "chunk_index": chunk_index,
                "ns": str(ns),
                "text": f"# {title}\n\n{infobox_data}",
            })
            chunk_index += 1
    except:
        pass
    
    try:
        sections = split_sections(text)
    except:
        sections = []
    
    skip_until_level = None
    level_titles = {}
    
    for sec_info in sections:
        section_title = clean_wiki_markup(sec_info['title'])
        section_level = sec_info['level']
        content = sec_info['content']

        if not content:
            continue
        
        if skip_until_level is not None:
            if section_level <= skip_until_level:
                skip_until_level = None
            else:
                continue
        
        if section_title.lower() in SKIP_SECTIONS:
            skip_until_level = section_level
            continue
        
        level_titles = {k: v for k, v in level_titles.items() if k < section_level}
        if section_level > 1 and section_title:
            level_titles[section_level] = section_title
        
        section_header = build_section_header(title, level_titles, section_level)
        
        def make_item(content_type, text_body):
            return {
                "document_id": make_doc_id(page_id, title, section_title, content_type, chunk_index),
                "page_id": str(page_id),
                "title": title,
                "url": url,
                "revid": str(revid),
                "language": 'en',
                "section": section_title,
                "section_level": section_level,
                "content_type": content_type,
                "chunk_index": chunk_index,
                "ns": str(ns),
                "text": section_header + text_body,
            }

        try:
            sec_text = extract_plain_text(content)
            if sec_text:
                write_item(out_file, make_item('sentence', sec_text))
                chunk_index += 1
        except Exception:
            pass

        try:
            for lst_text in extract_lists_from_content(content):
                if lst_text:
                    write_item(out_file, make_item('list', lst_text))
                    chunk_index += 1
        except Exception:
            pass

        try:
            for table_text in extract_tables_from_content(content):
                if table_text:
                    write_item(out_file, make_item('table', table_text))
                    chunk_index += 1
        except Exception:
            pass


def parse_xml_stream(input_bz2: str, output_jsonl: str) -> None:
    
    """
    Stream-parse a Wikipedia XML bz2 dump and write extracted content to a JSONL file.
    """

    global PAGE_COUNT, PROCESSED_COUNT, ITEM_COUNT
    
    with open(output_jsonl, 'w', encoding='utf-8') as out_file:
        with bz2.open(input_bz2, 'rt', encoding='utf-8') as f:
            context = ET.iterparse(f, events=('end',))
            
            page_data = {}
            skip_text = False
            
            for event, elem in context:
                tag = elem.tag.split('}')[-1]

                if tag == 'title':
                    page_data['title'] = elem.text
                elif tag == 'ns':
                    page_data['ns'] = elem.text
                elif tag == 'id':
                    if 'page_id' not in page_data:
                        page_data['page_id'] = elem.text
                        try:
                            skip_text = int(elem.text) < WORKER_START
                        except Exception:
                            skip_text = False
                    elif 'revid' not in page_data:
                        page_data['revid'] = elem.text
                elif tag == 'text':
                    if not skip_text:
                        page_data['text'] = elem.text or ''
                elif tag == 'page':
                    if not skip_text and all(k in page_data for k in ['page_id', 'title', 'ns', 'text']):
                        if should_process_page(page_data['page_id']):
                            PROCESSED_COUNT += 1
                            process_page(
                                page_data['page_id'],
                                page_data['title'],
                                page_data['ns'],
                                page_data.get('revid', ''),
                                page_data['text'],
                                out_file,
                            )

                    PAGE_COUNT += 1
                    current_page_id = int(page_data.get('page_id', 0))

                    if PAGE_COUNT % 1000 == 0:
                        elapsed = time.time() - START_TIME
                        pages_per_sec = PAGE_COUNT / elapsed if elapsed > 0 else 0

                        if current_page_id < WORKER_START:
                            sys.stdout.write(
                                f"\r[Worker {WORKER_ID}/{NUM_WORKERS}] SKIPPING → {WORKER_START:,} | "
                                f"current: {current_page_id:,} | scanned: {PAGE_COUNT:,} | "
                                f"{pages_per_sec:.0f} p/s   "
                            )
                        else:
                            sys.stdout.write(
                                f"\r[Worker {WORKER_ID}/{NUM_WORKERS}] "
                                f"page_id: {current_page_id:,} / {WORKER_END:,} | "
                                f"scanned: {PAGE_COUNT:,} | processed: {PROCESSED_COUNT:,} | "
                                f"items: {ITEM_COUNT:,} | {pages_per_sec:.0f} p/s   "
                            )
                        sys.stdout.flush()

                    if current_page_id >= WORKER_END:
                        print(f"\n[Worker {WORKER_ID}/{NUM_WORKERS}] "
                              f"Reached page_id {current_page_id:,} >= {WORKER_END:,}, stopping early.")
                        break

                    page_data = {}
                    skip_text = False
                    elem.clear()

    print()


def merge_corpus(output_dir: str) -> None:

    """
    Merge all corpus_N.jsonl files in output_dir into a single corpus.jsonl
    """

    input_files = sorted(glob.glob(f"{output_dir}/corpus_*.jsonl"))

    if not input_files:
        print("No worker files found to merge.")
        return

    print(f"Merging {len(input_files)} file(s)...")
    for f in input_files:
        print(f"  - {f}")

    all_items = []
    for filepath in input_files:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    all_items.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"  Warning: skipping invalid JSON at {filepath}:{line_num}: {e}")

    print(f"Sorting {len(all_items):,} items...")
    all_items.sort(key=lambda x: (int(x.get('page_id', 0)), x.get('chunk_index', 0)))

    output_path = f"{output_dir}/corpus.jsonl"
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in all_items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Merged {len(all_items):,} items → {output_path}")


if __name__ == '__main__': 

    """
    # parsing only
    python wikidump_parser.py --input dump.xml.bz2 --output-dir ./output 0 4

    # parsing + merging (worker 0)
    python wikidump_parser.py --input dump.xml.bz2 --output-dir ./output 0 1 --merge
    """

    parser = argparse.ArgumentParser(description='Wikipedia XML dump parser')
    parser.add_argument('worker_id', nargs='?', type=int, default=0, help='Worker ID (0-based)')
    parser.add_argument('num_workers', nargs='?', type=int, default=1, help='Total number of workers')
    parser.add_argument('--start-page-id', type=int, default=0, help='Start processing from this page_id (inclusive)')
    parser.add_argument('--end-page-id', type=int, default=82100000, help='Stop processing at this page_id (exclusive)')
    parser.add_argument('--input', required=True, help='Path to .xml.bz2 dump')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--merge', action='store_true', help='Merge corpus_N.jsonl files into corpus.jsonl after parsing')
    args = parser.parse_args()

    WORKER_ID = args.worker_id
    NUM_WORKERS = args.num_workers
    START_PAGE_ID = args.start_page_id
    END_PAGE_ID = args.end_page_id

    if WORKER_ID < 0 or WORKER_ID >= NUM_WORKERS:
        print(f"Error: worker_id ({WORKER_ID}) must be between 0 and {NUM_WORKERS - 1}")
        sys.exit(1)

    CHUNK_SIZE   = (END_PAGE_ID - START_PAGE_ID) // NUM_WORKERS
    WORKER_START = START_PAGE_ID + WORKER_ID * CHUNK_SIZE
    WORKER_END   = END_PAGE_ID if WORKER_ID == NUM_WORKERS - 1 else WORKER_START + CHUNK_SIZE

    OUTPUT_JSONL = (
        f"{args.output_dir}/corpus.jsonl" if NUM_WORKERS == 1
        else f"{args.output_dir}/corpus_{WORKER_ID}.jsonl"
    )

    PAGE_COUNT = 0
    PROCESSED_COUNT = 0
    ITEM_COUNT = 0
    START_TIME = time.time()

    print(f"[Worker {WORKER_ID}/{NUM_WORKERS}] Starting...")
    print(f"[Worker {WORKER_ID}/{NUM_WORKERS}] Input:  {args.input}")
    print(f"[Worker {WORKER_ID}/{NUM_WORKERS}] Output: {OUTPUT_JSONL}")
    print(f"[Worker {WORKER_ID}/{NUM_WORKERS}] Page ID range: {WORKER_START:,} <= page_id < {WORKER_END:,}")

    try:
        parse_xml_stream(args.input, OUTPUT_JSONL)
    except KeyboardInterrupt:
        print(f"\n[Worker {WORKER_ID}/{NUM_WORKERS}] Interrupted by user")
    except Exception as e:
        import traceback
        print(f"[Worker {WORKER_ID}/{NUM_WORKERS}] Error: {e}")
        traceback.print_exc()

    elapsed = time.time() - START_TIME
    print(f"\n[Worker {WORKER_ID}/{NUM_WORKERS}] Done!")
    print(f"  Scanned:   {PAGE_COUNT:,}")
    print(f"  Processed: {PROCESSED_COUNT:,}")
    print(f"  Items:     {ITEM_COUNT:,}")
    print(f"  Elapsed:   {elapsed:.1f}s")
    print(f"  Output:    {OUTPUT_JSONL}")

    if args.merge:
        merge_corpus(args.output_dir)
