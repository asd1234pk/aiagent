'''
Module to scrape and process articles from a website using its sitemap.xml.
'''
import os
import requests
from bs4 import BeautifulSoup # BeautifulSoup can also parse XML
import re
import xml.etree.ElementTree as ET
from urllib.parse import urljoin
import time # For rate limiting

# --- Configuration --- (請使用者自行修改)
SITEMAP_URL = "https://www.ohealth.com.tw/sitemap.xml"
WEBSITE_BASE_URL = "https://www.ohealth.com.tw" # 主要用於日誌或備用
SITEMAP_URL_FOR_STATUS_DISPLAY = SITEMAP_URL

# 文章 URL 的篩選關鍵字 (用於從 sitemap 中識別文章頁面)
# 例如，如果文章 URL 包含 '/posts/' 或 '/articles/'
ARTICLE_URL_KEYWORDS = ['/treatments/', '/showcases/', '/news/', '/faqs/', '/pain_guides/'] # 根據使用者需求更新
EXCLUDE_URL_FRAGMENTS = [] # 新增此行以解決 NameError

# CSS Selector for the main content of an article
# ARTICLE_CONTENT_SELECTOR = "main.main-content" # 預設值，可能需要根據網站結構修改
# ARTICLE_CONTENT_SELECTOR = "div.blog-details_content" # 上一個版本
ARTICLE_CONTENT_SELECTOR = "div.wrap[data-page=\'post-inner\']" # 使用者修正後的選擇器

# --- Text Splitting Configuration ---
CHUNK_SIZE_CHARS = 1000
CHUNK_OVERLAP_CHARS = 100

REQUEST_TIMEOUT = 15 # seconds, sitemap 可能較大
REQUEST_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

REQUEST_DELAY = 1 # Second(s) between requests to be polite to the server

def fetch_xml_or_html(url):
    '''Fetches XML or HTML content from a URL.'''
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT, headers=REQUEST_HEADERS)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        return None

def extract_links_from_sitemap(xml_content):
    '''Extracts and filters article links from sitemap.xml content.'''
    links = set()
    if not xml_content:
        return list(links)
    
    soup = BeautifulSoup(xml_content, 'xml') # 使用 'xml' 解析器
    loc_tags = soup.find_all('loc')
    
    print(f"DEBUG: Found {len(loc_tags)} <loc> tags in sitemap.")
    
    for tag in loc_tags:
        url = tag.string.strip() if tag.string else None
        if url:
            # 篩選邏輯：
            # 1. 必須包含 ARTICLE_PATH_KEYWORDS 中的任一關鍵字
            # 2. 不能是 EXCLUDE_URL_FRAGMENTS 中的任一個
            # 3. 簡單 heuristic: URL 路徑深度通常較深 (例如多於3個'/') 才可能是具體文章頁
            path_parts = url.replace(WEBSITE_BASE_URL, '').strip('/').split('/')
            
            is_article_path = any(keyword in url for keyword in ARTICLE_URL_KEYWORDS)
            is_excluded = any(excluded_fragment in url for excluded_fragment in EXCLUDE_URL_FRAGMENTS)
            
            # 特別處理完全匹配的排除項
            if url in EXCLUDE_URL_FRAGMENTS:
                 # print(f"DEBUG: Skipping excluded URL (exact match): {url}")
                 continue

            # 篩選掉主分類頁，只保留更深層級的文章頁
            # 例如 /showcases/client-testimonials-shoulderandneck 可能是分類，而 /showcases/client-testimonials-shoulderandneck/lao-zhen-taoyuan 是文章
            # 這裡用一個簡單的判斷：如果 URL 以 /showcases 或 /pain_guides 結尾，且不再有下一級，則可能是分類頁
            # is_likely_category_page = False
            # if (url.endswith("/showcases") or url.endswith("/pain_guides") or url.endswith("/news")):
            #      is_likely_category_page = True
            # for keyword in ARTICLE_PATH_KEYWORDS:
            #     if url.startswith(WEBSITE_BASE_URL + keyword.rstrip('/')) and len(path_parts) <= len(keyword.strip('/').split('/')) + 1:
            #         # +1 是因為有些分類可能是 /showcases/some-category (兩層)
            #         if len(path_parts) <= 2: # e.g. /showcases/client-testimonials-waistandback
            #             is_likely_category_page = True
            #             # print(f"DEBUG: Likely category page (short path): {url}")
            #             break
            
            # 更精確的排除：檢查 URL 是否與 EXCLUDE_URL_FRAGMENTS 中的某個項目完全相同
            if url in EXCLUDE_URL_FRAGMENTS:
                # print(f"DEBUG: Skipping excluded URL: {url}")
                continue
            
            # 修正：確保 EXCLUDE_URL_FRAGMENTS 內的項目如果本身也是一個 path keyword 的根，不會被錯誤地排除掉所有子頁面
            # is_explicitly_excluded = url in EXCLUDE_URL_FRAGMENTS
            # is_sub_path_of_exclusion_but_not_deep_enough = False
            # for excluded in EXCLUDE_URL_FRAGMENTS:
            #     if url.startswith(excluded) and len(path_parts) == len(excluded.replace(WEBSITE_BASE_URL, '').strip('/').split('/')):
            #         is_sub_path_of_exclusion_but_not_deep_enough = True
            #         break
            
            if is_article_path and len(path_parts) > 1: # 確保不是只有 base URL + keyword 的情況 (例如 https://ohealth.com.tw/showcases/)
                                                        # 且路徑至少有兩部分 (如 /showcases/article-name)
                # print(f"DEBUG: Adding URL from sitemap: {url}")
                links.add(url)
            # else:
                # if not is_article_path:
                    # print(f"DEBUG: Skipping URL (not article path): {url}")
                # elif is_likely_category_page:
                    # print(f"DEBUG: Skipping URL (likely category): {url}")
                # elif len(path_parts) <=1:
                    # print(f"DEBUG: Skipping URL (path too short): {url}")

    # print(f"DEBUG: Links after initial sitemap processing and filtering: {len(links)}")
    return list(links)

def extract_article_text_and_title(html_content, article_url):
    '''Extracts the main text content and title from an article page.'''
    if not html_content:
        return None, None

    soup = BeautifulSoup(html_content, 'html.parser')
    title = "Unknown Title"
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    elif soup.find('h1'):
        title = soup.find('h1').get_text(separator=' ', strip=True)

    content_element = soup.select_one(ARTICLE_CONTENT_SELECTOR)
    if content_element:
        for unwanted_tag in content_element.find_all(['script', 'style', 'nav', 'footer', 'aside', '.blog-author', '.blog-pagination', '.sidebar', '.comments-area', '.share-links', '.blog-meta']): # 增加更多可能的排除項
            unwanted_tag.decompose()
        text_parts = []
        for element in content_element.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'span', 'strong', 'em']):
            element_text = element.get_text(separator=' ', strip=True)
            if element_text:
                text_parts.append(element_text)
        raw_text = "\n".join(text_parts)
        cleaned_text = re.sub(r'\s{2,}', ' ', raw_text)
        cleaned_text = re.sub(r'\n{2,}', '\n', cleaned_text).strip()
        return cleaned_text, title
    else:
        print(f"Warning: Main content not found on {article_url} using selector '{ARTICLE_CONTENT_SELECTOR}'. Please check the selector for this page structure.")
        # print(f"DEBUG: HTML content for {article_url} (first 1000 chars):\n{html_content[:1000] if html_content else 'None'}")
        return None, title

def split_text_into_chunks(text, source_url, title):
    '''Splits text into manageable chunks.'''
    if not text:
        return []
    chunks = []
    start_index = 0
    doc_len = len(text)
    while start_index < doc_len:
        end_index = min(start_index + CHUNK_SIZE_CHARS, doc_len)
        if end_index < doc_len:
            possible_split_points = [i for i, char in enumerate(text[end_index:min(end_index + 100, doc_len)]) if char in '.!?\n']
            if possible_split_points:
                end_index = end_index + possible_split_points[0] + 1
        chunk_text = text[start_index:end_index]
        chunks.append({
            "text": chunk_text, "source": source_url, "title": title,
            "type": "website_article", "chunk_char_start": start_index, "chunk_char_end": end_index
        })
        if end_index == doc_len: break
        start_index = max(0, end_index - CHUNK_OVERLAP_CHARS)
        if start_index >= end_index: start_index = end_index
    return chunks

def process_website_articles(sitemap_url_to_process=SITEMAP_URL):
    '''
    Fetches article URLs from sitemap.xml, then scrapes each article.
    '''
    print(f"Starting website processing using sitemap: {sitemap_url_to_process}")
    all_text_chunks = []
    
    sitemap_xml = fetch_xml_or_html(sitemap_url_to_process)
    if not sitemap_xml:
        print(f"Failed to fetch sitemap: {sitemap_url_to_process}. Aborting website processing.")
        return all_text_chunks

    article_urls = extract_links_from_sitemap(sitemap_xml)
    if not article_urls:
        print(f"No valid article URLs extracted from sitemap {sitemap_url_to_process}. Check filtering logic in extract_links_from_sitemap.")
        return all_text_chunks
    
    print(f"Found {len(article_urls)} potential article URLs from sitemap to process.")

    for i, url in enumerate(article_urls):
        print(f"Processing article {i+1}/{len(article_urls)}: {url}")
        article_html = fetch_xml_or_html(url)
        if article_html:
            article_text, article_title = extract_article_text_and_title(article_html, url)
            if article_text:
                chunks = split_text_into_chunks(article_text, url, article_title)
                if chunks:
                    all_text_chunks.extend(chunks)
                    print(f"  -> Extracted {len(chunks)} chunks from '{article_title}' ({url})")
                else:
                    print(f"  -> No text chunks extracted from '{article_title}' ({url}).")
            else:
                print(f"  -> Failed to extract text content from '{article_title}' ({url}). Check ARTICLE_CONTENT_SELECTOR or website structure for: {url}")
        else:
            print(f"  -> Failed to fetch HTML for article: {url}")
            
    print(f"Total text chunks extracted from website articles via sitemap: {len(all_text_chunks)}")
    return all_text_chunks

# --- Main execution for testing ---
if __name__ == '__main__':
    print("--- Testing Website Scraper (Sitemap Method) ---")
    
    # 關鍵：確認 ARTICLE_CONTENT_SELECTOR 是否適用於 sitemap 中找到的文章頁面結構。
    # 您可能需要檢查幾條從 sitemap 來的文章，確認它們的 HTML 結構，並相應調整 ARTICLE_CONTENT_SELECTOR。
    if ARTICLE_CONTENT_SELECTOR == "article.main-content-article": # 如果還是預設的範例 selector
        print("\nWARNING: ARTICLE_CONTENT_SELECTOR is using a default example value ('article.main-content-article').")
        print("Please inspect a few article pages from the sitemap (e.g., under /showcases/ or /pain_guides/) and update ARTICLE_CONTENT_SELECTOR in the script to accurately target the main content area of those pages.")
        print("The current value is likely incorrect for ohealth.com.tw. It was previously suggested to be 'div.blog-details_content'. Please verify and set it correctly.")

    processed_chunks = process_website_articles()
    if processed_chunks:
        print(f"\nSuccessfully processed {len(processed_chunks)} chunks from the website via sitemap.")
        # print("First few chunks:")
        # for i, chunk in enumerate(processed_chunks[:min(3, len(processed_chunks))]):
        #     print(f"--- Chunk {i+1} (Source: {chunk['source']}, Title: {chunk['title']}) ---")
        #     print(chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"])
        #     print(f"Char start: {chunk['chunk_char_start\']}, Char end: {chunk[\'chunk_char_end\']}\\n")
    else:
        print("No chunks were processed from the website via sitemap. Check configurations, sitemap accessibility, and ARTICLE_CONTENT_SELECTOR.")
    print("--- Website Scraper Test Complete (Sitemap Method) ---") 