"""
Claw Scraper API v3.1 - NVIDIA Edition
Production FastAPI backend with NVIDIA API for AI features
"""

import os
import json
import asyncio
import hashlib
import re
import time
import random
from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from urllib.parse import urljoin, urlparse

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Scraping
import requests
from bs4 import BeautifulSoup, Comment

# AI - NVIDIA NIM API (OpenAI-compatible)
try:
    from openai import AsyncOpenAI
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

# Token counting fallback
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

# Distributed
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Browser
try:
    from playwright.async_api import async_playwright
    BROWSER_AVAILABLE = True
except ImportError:
    BROWSER_AVAILABLE = False

# Data processing
try:
    from markdownify import markdownify as md
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False

# ==================== CONFIGURATION ====================

class Config:
    # NVIDIA API Configuration
    NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
    NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
    NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "meta/llama-3.1-70b-instruct")

    # Legacy OpenAI support (optional)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    # AI Provider: "nvidia" or "openai"
    AI_PROVIDER = os.getenv("AI_PROVIDER", "nvidia").lower()

    # Redis
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # Proxies
    PROXY_LIST = [p.strip() for p in os.getenv("PROXY_LIST", "").split(",") if p.strip()]

    # Performance
    MAX_CONCURRENT_SCRAPES = int(os.getenv("MAX_CONCURRENT_SCRAPES", "5"))
    RATE_LIMIT_DELAY = float(os.getenv("RATE_LIMIT_DELAY", "1.0"))
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

config = Config()

# ==================== DATA MODELS ====================

class ScrapeRequest(BaseModel):
    url: str = Field(..., description="URL to scrape")
    render_js: bool = Field(False, description="Render JavaScript with browser")
    ai_extract: bool = Field(False, description="Use AI to extract structured data")
    ai_schema: Optional[Dict[str, str]] = Field(None, description="Schema for AI extraction")
    detect_schema: bool = Field(False, description="Auto-detect extraction schema")
    summarize: bool = Field(False, description="Generate AI summary")
    proxy_enabled: bool = Field(True, description="Use proxy rotation")

class BatchScrapeRequest(BaseModel):
    urls: List[str] = Field(..., description="List of URLs to scrape", max_length=50)
    render_js: bool = Field(False)
    ai_extract: bool = Field(False)
    ai_schema: Optional[Dict[str, str]] = None
    detect_schema: bool = Field(False)
    summarize: bool = Field(False)
    workers: int = Field(3, ge=1, le=10)

class AIQueryRequest(BaseModel):
    content: str = Field(..., description="Content to analyze")
    query_type: str = Field("extract", description="extract|summarize|classify|entities|qa")
    schema: Optional[Dict[str, str]] = None
    categories: Optional[List[str]] = None
    num_questions: int = Field(5, ge=1, le=10)

class ScrapedContent(BaseModel):
    url: str
    title: str
    content: str
    markdown: str
    metadata: Dict[str, Any]
    links: List[str]
    images: List[Dict[str, str]]
    timestamp: str
    word_count: int
    ai_extracted: Dict[str, Any] = {}
    detected_schema: Dict[str, Any] = {}
    proxy_used: Optional[str] = None
    crawl_depth: int = 0
    status: str = "success"
    error: Optional[str] = None

# ==================== PROXY ROTATION ====================

class ProxyConfig:
    def __init__(self, host, port, username=None, password=None, protocol="http", weight=1):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.protocol = protocol
        self.weight = weight
        self.failures = 0
        self.last_used = 0

    @property
    def url(self):
        if self.username and self.password:
            return f"{self.protocol}://{self.username}:{self.password}@{self.host}:{self.port}"
        return f"{self.protocol}://{self.host}:{self.port}"

    @property
    def dict_format(self):
        return {"http": self.url, "https": self.url}

class ProxyRotator:
    def __init__(self, proxies, strategy="weighted", max_failures=3):
        self.proxies = []
        self.strategy = strategy
        self.max_failures = max_failures
        self._current_index = 0

        for proxy in proxies:
            if isinstance(proxy, str) and proxy.strip():
                self.proxies.append(self._parse_proxy_string(proxy))

    def _parse_proxy_string(self, proxy_str):
        pattern = r"(?:(?P<protocol>https?)://)?(?:(?P<user>[^:]+):(?P<pass>[^@]+)@)?(?P<host>[^:]+):(?P<port>\d+)"
        match = re.match(pattern, proxy_str.strip())
        if not match:
            return ProxyConfig(host="localhost", port=8080)
        return ProxyConfig(
            host=match.group("host"),
            port=int(match.group("port")),
            username=match.group("user"),
            password=match.group("pass"),
            protocol=match.group("protocol") or "http"
        )

    def get_proxy(self):
        if not self.proxies:
            return None
        healthy = [p for p in self.proxies if p.failures < self.max_failures]
        if not healthy:
            for p in self.proxies:
                p.failures = 0
            healthy = self.proxies

        if self.strategy == "round_robin":
            proxy = healthy[self._current_index % len(healthy)]
            self._current_index += 1
        elif self.strategy == "weighted":
            weights = [p.weight for p in healthy]
            proxy = random.choices(healthy, weights=weights, k=1)[0]
        else:
            proxy = random.choice(healthy)

        proxy.last_used = time.time()
        return proxy

    def report_failure(self, proxy):
        proxy.failures += 1

    def report_success(self, proxy):
        if proxy.failures > 0:
            proxy.failures = max(0, proxy.failures - 1)

    def get_stats(self):
        return {
            "total": len(self.proxies),
            "healthy": len([p for p in self.proxies if p.failures < self.max_failures]),
            "proxies": [{"host": p.host, "port": p.port, "failures": p.failures, "healthy": p.failures < self.max_failures} for p in self.proxies]
        }

# ==================== AI EXTRACTOR (NVIDIA + OPENAI) ====================

class AIExtractor:
    """
    Unified AI extractor supporting both NVIDIA NIM and OpenAI APIs.
    NVIDIA NIM uses OpenAI-compatible endpoints.
    """

    def __init__(self, api_key=None, base_url=None, model=None, provider="nvidia"):
        if not AI_AVAILABLE:
            raise ImportError("Install openai: pip install openai")

        self.provider = provider

        if provider == "nvidia":
            self.api_key = api_key or config.NVIDIA_API_KEY
            self.base_url = base_url or config.NVIDIA_BASE_URL
            self.model = model or config.NVIDIA_MODEL
        else:
            self.api_key = api_key or config.OPENAI_API_KEY
            self.base_url = base_url or config.OPENAI_BASE_URL
            self.model = model or "gpt-4o-mini"

        if not self.api_key:
            raise ValueError(f"No API key provided for {provider}. Set NVIDIA_API_KEY or OPENAI_API_KEY")

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        # Token counting
        if TIKTOKEN_AVAILABLE and provider == "openai":
            try:
                self.tokenizer = tiktoken.encoding_for_model("gpt-4")
            except:
                self.tokenizer = None
        else:
            self.tokenizer = None

    def _count_tokens(self, text):
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        # Rough estimate for NVIDIA models
        return len(text.split()) * 1.3

    def _truncate(self, text, max_tokens=6000):
        if self.tokenizer:
            tokens = self.tokenizer.encode(text)
            if len(tokens) <= max_tokens:
                return text
            return self.tokenizer.decode(tokens[:max_tokens])
        # Fallback for NVIDIA
        words = text.split()
        if len(words) <= max_tokens:
            return text
        return " ".join(words[:max_tokens])

    async def _call_llm(self, system_prompt, user_prompt, max_tokens=4000, temperature=0.1, json_mode=True):
        """Call LLM with provider-specific handling"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        # NVIDIA supports response_format for some models
        if json_mode and "llama" in self.model.lower():
            # Llama models may not support response_format, handle gracefully
            try:
                kwargs["response_format"] = {"type": "json_object"}
                response = await self.client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                # Try to parse as JSON
                try:
                    return json.loads(content)
                except:
                    # Extract JSON from text if wrapped in markdown
                    json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group(1))
                    # Try to find JSON object
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group(0))
                    return {"result": content}
            except Exception as e:
                # Fallback: request JSON explicitly in prompt
                return await self._call_llm_json_fallback(system_prompt, user_prompt, max_tokens, temperature)
        else:
            response = await self.client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            try:
                return json.loads(content)
            except:
                return {"result": content}

    async def _call_llm_json_fallback(self, system_prompt, user_prompt, max_tokens, temperature):
        """Fallback for models that don't support response_format"""
        enhanced_prompt = user_prompt + "\n\nIMPORTANT: Return ONLY a valid JSON object. No markdown, no explanations, just JSON."

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": enhanced_prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )

        content = response.choices[0].message.content
        try:
            return json.loads(content)
        except:
            # Extract JSON
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except:
                    pass
            return {"raw_response": content, "parse_error": True}

    async def extract_structured(self, content, schema):
        schema_desc = "\n".join([f"- {k}: {v}" for k, v in schema.items()])
        system_prompt = f"""You are an expert data extraction AI. Extract structured information from the provided content.
Return ONLY a valid JSON object with these fields:
{schema_desc}
Rules:
- Extract exactly what is asked, no extra fields
- Use null if information is not found
- For lists, return arrays
- For dates, use ISO 8601 format"""

        try:
            result = await self._call_llm(
                system_prompt,
                f"Content to analyze:\n\n{self._truncate(content)}",
                max_tokens=4000,
                temperature=0.1
            )
            return {"success": True, "data": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def summarize(self, content, style="concise"):
        styles = {
            "concise": "Provide a brief 2-3 sentence summary.",
            "detailed": "Provide a comprehensive paragraph summary.",
            "bullet_points": "Provide 3-5 bullet points summarizing key information."
        }

        system_prompt = "You are a summarization expert. Return ONLY valid JSON."
        prompt = f"""Summarize the following content. {styles.get(style, styles["concise"])}
Content:
{self._truncate(content)}
Return JSON: {{"summary": "your summary here", "key_points": ["point1", "point2"]}}"""

        try:
            return await self._call_llm(system_prompt, prompt, max_tokens=1000, temperature=0.3)
        except Exception as e:
            return {"error": str(e)}

    async def classify(self, content, categories):
        system_prompt = "You are a content classification expert. Return ONLY valid JSON."
        prompt = f"""Classify this content into one of these categories: {', '.join(categories)}
Content:
{self._truncate(content)}
Return JSON: {{"category": "best matching category", "confidence": 0.95, "reasoning": "brief explanation"}}"""

        try:
            return await self._call_llm(system_prompt, prompt, max_tokens=500, temperature=0.1)
        except Exception as e:
            return {"error": str(e)}

    async def extract_entities(self, content):
        system_prompt = "Extract named entities accurately. Return ONLY valid JSON."
        prompt = f"""Extract named entities from this content.
Content:
{self._truncate(content)}
Return JSON: {{
    "people": ["name1", "name2"],
    "organizations": ["org1", "org2"],
    "locations": ["location1"],
    "dates": ["2024-01-01"],
    "products": ["product1"],
    "technologies": ["tech1"]
}}"""

        try:
            return await self._call_llm(system_prompt, prompt, max_tokens=1000, temperature=0.1)
        except Exception as e:
            return {"error": str(e)}

    async def generate_qa(self, content, num_questions=5):
        system_prompt = "Generate educational Q&A pairs. Return ONLY valid JSON."
        prompt = f"""Generate {num_questions} question-answer pairs based on this content.
Content:
{self._truncate(content)}
Return JSON: {{"qa_pairs": [{{"question": "...", "answer": "...", "difficulty": "easy|medium|hard"}}]}}"""

        try:
            result = await self._call_llm(system_prompt, prompt, max_tokens=2000, temperature=0.3)
            return result.get("qa_pairs", [])
        except Exception as e:
            return [{"error": str(e)}]

    async def detect_schema(self, html, url):
        soup = BeautifulSoup(html, 'html.parser')
        structure = self._get_structure(soup)
        text_sample = soup.get_text(separator=' ', strip=True)[:2000]

        system_prompt = "You are a web scraping schema designer. Return ONLY valid JSON."
        prompt = f"""Analyze this webpage and suggest extraction schema.
URL: {url}
Structure:
{structure}
Text Sample:
{text_sample}
Return JSON:
{{
    "content_type": "detected type (article/product/listing/profile/etc)",
    "schema": {{
        "field_name": {{
            "css_selector": "suggested CSS selector",
            "description": "what this field contains",
            "data_type": "text|number|date|url|list|image",
            "required": true/false
        }}
    }},
    "confidence": 0.0-1.0
}}"""

        try:
            return await self._call_llm(system_prompt, prompt, max_tokens=2000, temperature=0.1)
        except Exception as e:
            return {"error": str(e)}

    def _get_structure(self, soup):
        lines = []
        for meta in soup.find_all('meta', limit=10):
            name = meta.get('name', meta.get('property', ''))
            content = meta.get('content', '')
            if name and content:
                lines.append(f"META {name}: {content[:100]}")
        for tag in ['h1', 'h2', 'article', 'main']:
            for el in soup.find_all(tag, limit=3):
                lines.append(f"<{tag}>: {el.get_text(strip=True)[:100]}")
        return "\n".join(lines[:30])


# ==================== CORE SCRAPER ====================

class CoreScraper:
    def __init__(self):
        self.proxy_rotator = ProxyRotator(config.PROXY_LIST) if config.PROXY_LIST else None
        self.ai_extractor = None

        # Initialize AI based on provider
        if AI_AVAILABLE:
            try:
                if config.AI_PROVIDER == "nvidia" and config.NVIDIA_API_KEY:
                    self.ai_extractor = AIExtractor(
                        api_key=config.NVIDIA_API_KEY,
                        base_url=config.NVIDIA_BASE_URL,
                        model=config.NVIDIA_MODEL,
                        provider="nvidia"
                    )
                elif config.OPENAI_API_KEY:
                    self.ai_extractor = AIExtractor(
                        api_key=config.OPENAI_API_KEY,
                        base_url=config.OPENAI_BASE_URL,
                        provider="openai"
                    )
            except Exception as e:
                print(f"AI initialization failed: {e}")

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "ClawBot/3.1 (Research Purpose)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        })
        self._semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_SCRAPES)
        self._last_request = 0

    async def _rate_limit(self):
        async with self._semaphore:
            elapsed = time.time() - self._last_request
            delay = config.RATE_LIMIT_DELAY + random.uniform(0, 0.5)
            if elapsed < delay:
                await asyncio.sleep(delay - elapsed)
            self._last_request = time.time()

    def _clean_html(self, soup):
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'form', 'button', 'input', 'iframe', 'noscript']):
            element.decompose()
        for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
            comment.extract()
        for cls in ['advertisement', 'ad-', 'ads-', 'social-share', 'comments', 'sidebar', 'widget', 'popup', 'cookie']:
            for el in soup.find_all(class_=re.compile(cls, re.I)):
                el.decompose()
        return soup

    def _extract_main(self, soup):
        for tag in ['article', 'main']:
            content = soup.find(tag)
            if content:
                return str(content)
        candidates = soup.find_all(['div', 'section'])
        best = None
        best_score = 0
        for c in candidates:
            text = c.get_text(strip=True)
            paras = len(c.find_all('p'))
            score = len(text) + paras * 100
            if score > best_score:
                best_score = score
                best = c
        return str(best) if best else str(soup.body or soup)

    def _extract_meta(self, soup, url):
        meta = {"url": url, "scraped_at": datetime.now().isoformat()}
        selectors = {
            "title": ["og:title", "twitter:title"],
            "description": ["og:description", "twitter:description", "description"],
            "image": ["og:image", "twitter:image"],
            "author": ["author"],
            "published_time": ["article:published_time"],
        }
        for key, props in selectors.items():
            for prop in props:
                tag = soup.find("meta", property=prop) or soup.find("meta", attrs={"name": prop})
                if tag and tag.get("content"):
                    meta[key] = tag["content"]
                    break
        if "title" not in meta:
            t = soup.find("title")
            if t:
                meta["title"] = t.get_text(strip=True)
        return meta

    async def _fetch(self, url, render_js=False, use_proxy=True):
        await self._rate_limit()
        proxy = None
        if use_proxy and self.proxy_rotator:
            proxy = self.proxy_rotator.get_proxy()
        proxies = proxy.dict_format if proxy else None

        try:
            if render_js and BROWSER_AVAILABLE:
                return await self._render_js(url, proxy)
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.session.get(url, proxies=proxies, timeout=config.REQUEST_TIMEOUT, allow_redirects=True)
            )
            response.raise_for_status()
            if proxy:
                self.proxy_rotator.report_success(proxy)
            return response.text
        except Exception as e:
            if proxy:
                self.proxy_rotator.report_failure(proxy)
            raise e

    async def _render_js(self, url, proxy=None):
        if not BROWSER_AVAILABLE:
            raise ImportError("Playwright not installed")
        proxy_config = None
        if proxy:
            proxy_config = {"server": f"{proxy.protocol}://{proxy.host}:{proxy.port}"}
            if proxy.username:
                proxy_config["username"] = proxy.username
                proxy_config["password"] = proxy.password
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, proxy=proxy_config)
            page = await browser.new_page()
            await page.goto(url, wait_until="networkidle", timeout=config.REQUEST_TIMEOUT * 1000)
            html = await page.content()
            await browser.close()
            return html

    async def scrape(self, url, options):
        try:
            html = await self._fetch(url, options.render_js, options.proxy_enabled)
            soup = BeautifulSoup(html, 'html.parser')
            clean = self._clean_html(soup)
            main_html = self._extract_main(clean)
            meta = self._extract_meta(soup, url)
            title = meta.get("title", "Untitled")
            content = BeautifulSoup(main_html, 'html.parser').get_text(separator='\n', strip=True)
            markdown = md(main_html, heading_style="ATX") if MARKDOWN_AVAILABLE else content

            base = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
            links = []
            for a in soup.find_all('a', href=True):
                full = urljoin(base, a['href'])
                if urlparse(full).scheme in ('http', 'https'):
                    links.append(full)
            images = []
            for img in soup.find_all('img', src=True):
                images.append({"url": urljoin(base, img['src']), "alt": img.get("alt", "")})

            ai_data = {}
            detected_schema = {}

            if options.detect_schema and self.ai_extractor:
                detected_schema = await self.ai_extractor.detect_schema(html, url)

            if options.ai_extract and self.ai_extractor:
                schema = options.ai_schema
                if not schema and detected_schema and "schema" in detected_schema:
                    schema = {k: v["description"] for k, v in detected_schema["schema"].items()}
                if schema:
                    extraction = await self.ai_extractor.extract_structured(content, schema)
                    if extraction.get("success"):
                        ai_data["extracted"] = extraction["data"]

            if options.summarize and self.ai_extractor:
                ai_data["summary"] = await self.ai_extractor.summarize(content)

            proxy_info = None
            if self.proxy_rotator:
                stats = self.proxy_rotator.get_stats()
                proxy_info = f"{stats['healthy']}/{stats['total']} healthy"

            return ScrapedContent(
                url=url, title=title, content=content, markdown=markdown,
                metadata=meta, links=list(set(links))[:50], images=images[:20],
                timestamp=datetime.now().isoformat(), word_count=len(content.split()),
                ai_extracted=ai_data, detected_schema=detected_schema,
                proxy_used=proxy_info
            )
        except Exception as e:
            return ScrapedContent(
                url=url, title="Error", content="", markdown="",
                metadata={}, links=[], images=[], timestamp=datetime.now().isoformat(),
                word_count=0, status="error", error=str(e)
            )

    async def batch_scrape(self, urls, options):
        req = ScrapeRequest(
            url="", render_js=options.render_js, ai_extract=options.ai_extract,
            ai_schema=options.ai_schema, detect_schema=options.detect_schema,
            summarize=options.summarize
        )
        tasks = [self.scrape(url, req) for url in urls]
        return await asyncio.gather(*tasks)

# ==================== REDIS DISTRIBUTED ====================

class DistributedManager:
    def __init__(self):
        self.available = False
        if REDIS_AVAILABLE:
            try:
                self.redis = redis.from_url(config.REDIS_URL, decode_responses=True)
                self.redis.ping()
                self.available = True
            except:
                self.redis = None

    def _hash(self, url):
        return hashlib.sha256(url.encode()).hexdigest()[:16]

    def enqueue(self, urls, queue_name="claw_default"):
        if not self.available:
            return 0
        added = 0
        for url in urls:
            h = self._hash(url)
            if not self.redis.sismember(f"{queue_name}:visited", h):
                task = json.dumps({"url": url, "enqueued_at": datetime.now().isoformat(), "attempts": 0})
                self.redis.zadd(f"{queue_name}:pending", {task: 5})
                added += 1
        return added

    def get_stats(self, queue_name="claw_default"):
        if not self.available:
            return {"available": False}
        return {
            "available": True,
            "pending": self.redis.zcard(f"{queue_name}:pending"),
            "visited": self.redis.scard(f"{queue_name}:visited"),
            "completed": int(self.redis.hget(f"{queue_name}:stats", "completed") or 0),
            "failed": int(self.redis.hget(f"{queue_name}:stats", "failed") or 0)
        }

    def get_results(self, queue_name="claw_default", limit=100):
        if not self.available:
            return []
        results = []
        for key in self.redis.keys(f"{queue_name}:results:*")[:limit]:
            data = self.redis.get(key)
            if data:
                results.append(json.loads(data))
        return results

# ==================== FASTAPI APP ====================

scraper = CoreScraper()
dist_manager = DistributedManager()

app = FastAPI(
    title="Claw Scraper API - NVIDIA Edition",
    description="Advanced web scraping with NVIDIA AI, proxies, and distributed crawling",
    version="3.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
async def health_check():
    ai_info = {}
    if scraper.ai_extractor:
        ai_info = {
            "configured": True,
            "provider": scraper.ai_extractor.provider,
            "model": scraper.ai_extractor.model,
            "base_url": scraper.ai_extractor.base_url
        }
    else:
        ai_info = {"configured": False, "provider": config.AI_PROVIDER}

    return {
        "status": "healthy",
        "ai": ai_info,
        "proxies_configured": scraper.proxy_rotator is not None,
        "proxies_count": len(scraper.proxy_rotator.proxies) if scraper.proxy_rotator else 0,
        "redis_available": dist_manager.available,
        "browser_available": BROWSER_AVAILABLE,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/proxies/stats")
async def proxy_stats():
    if not scraper.proxy_rotator:
        return {"configured": False}
    return {"configured": True, **scraper.proxy_rotator.get_stats()}

@app.post("/api/scrape")
async def api_scrape(request: ScrapeRequest):
    result = await scraper.scrape(request.url, request)
    if result.status == "error":
        raise HTTPException(status_code=400, detail=result.error)
    return result

@app.post("/api/batch")
async def api_batch(request: BatchScrapeRequest):
    results = await scraper.batch_scrape(request.urls, request)
    successful = [r for r in results if r.status == "success"]
    failed = [r for r in results if r.status == "error"]
    return {
        "total": len(request.urls),
        "successful": len(successful),
        "failed": len(failed),
        "results": [{"url": r.url, "title": r.title, "status": r.status, "word_count": r.word_count, "links": r.links, "error": r.error} for r in results]
    }

@app.post("/api/ai/query")
async def ai_query(request: AIQueryRequest):
    if not scraper.ai_extractor:
        raise HTTPException(status_code=503, detail=f"AI not configured. Set NVIDIA_API_KEY (provider: {config.AI_PROVIDER})")

    result = {}
    if request.query_type == "extract":
        if not request.schema:
            raise HTTPException(status_code=400, detail="Schema required for extraction")
        result = await scraper.ai_extractor.extract_structured(request.content, request.schema)
    elif request.query_type == "summarize":
        result = await scraper.ai_extractor.summarize(request.content)
    elif request.query_type == "classify":
        if not request.categories:
            raise HTTPException(status_code=400, detail="Categories required for classification")
        result = await scraper.ai_extractor.classify(request.content, request.categories)
    elif request.query_type == "entities":
        result = await scraper.ai_extractor.extract_entities(request.content)
    elif request.query_type == "qa":
        result = await scraper.ai_extractor.generate_qa(request.content, request.num_questions)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown query type: {request.query_type}")

    return {"query_type": request.query_type, "provider": scraper.ai_extractor.provider, "model": scraper.ai_extractor.model, "result": result}

@app.post("/api/distributed/enqueue")
async def dist_enqueue(urls: List[str]):
    if not dist_manager.available:
        raise HTTPException(status_code=503, detail="Redis not available")
    count = dist_manager.enqueue(urls)
    return {"enqueued": count, "urls": urls}

@app.get("/api/distributed/stats")
async def dist_stats():
    return dist_manager.get_stats()

@app.get("/api/distributed/results")
async def dist_results(limit: int = 100):
    if not dist_manager.available:
        raise HTTPException(status_code=503, detail="Redis not available")
    return dist_manager.get_results(limit=limit)


# ==================== MOBILE HTML UI ====================

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <meta name="theme-color" content="#76b900">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <title>Claw Scraper - NVIDIA AI</title>
    <style>
        :root {
            --primary: #76b900; --primary-dark: #5a8f00;
            --success: #10b981; --warning: #f59e0b;
            --danger: #ef4444; --dark: #0f172a;
            --gray: #64748b; --light: #f1f5f9;
            --bg: #f8fafc; --card: #ffffff;
            --radius: 16px; --shadow: 0 4px 20px rgba(0,0,0,0.08);
        }
        * { margin: 0; padding: 0; box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg); color: var(--dark); line-height: 1.5;
            -webkit-font-smoothing: antialiased; overflow-x: hidden;
        }
        .app-container { max-width: 100%; min-height: 100vh; padding-bottom: 80px; }

        .header {
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
            color: white; padding: 20px 16px 24px; position: sticky; top: 0; z-index: 100;
            backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
        }
        .header-top { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; }
        .logo { font-size: 1.5em; font-weight: 800; letter-spacing: -0.5px; }
        .version { font-size: 0.7em; opacity: 0.8; background: rgba(255,255,255,0.2); padding: 2px 8px; border-radius: 20px; }
        .header-subtitle { font-size: 0.85em; opacity: 0.9; }

        .status-bar {
            display: flex; gap: 8px; padding: 12px 16px; overflow-x: auto;
            -webkit-overflow-scrolling: touch; scrollbar-width: none;
        }
        .status-bar::-webkit-scrollbar { display: none; }
        .status-pill {
            padding: 6px 12px; border-radius: 20px; font-size: 0.75em; font-weight: 600;
            white-space: nowrap; display: flex; align-items: center; gap: 4px;
        }
        .status-pill.online { background: #d1fae5; color: #065f46; }
        .status-pill.offline { background: #fee2e2; color: #991b1b; }
        .status-pill.warning { background: #fef3c7; color: #92400e; }
        .status-dot { width: 6px; height: 6px; border-radius: 50%; display: inline-block; }
        .status-dot.active { background: currentColor; animation: pulse 2s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }

        .bottom-nav {
            position: fixed; bottom: 0; left: 0; right: 0;
            background: rgba(255,255,255,0.95); backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px); border-top: 1px solid rgba(0,0,0,0.05);
            display: flex; justify-content: space-around; padding: 8px 0 calc(8px + env(safe-area-inset-bottom));
            z-index: 1000;
        }
        .nav-item {
            display: flex; flex-direction: column; align-items: center; gap: 2px;
            padding: 4px 12px; border: none; background: none; color: var(--gray);
            font-size: 0.65em; font-weight: 600; cursor: pointer; transition: all 0.2s;
        }
        .nav-item.active { color: var(--primary); }
        .nav-item .icon { font-size: 1.6em; line-height: 1; }

        .section { display: none; padding: 16px; animation: fadeIn 0.3s ease; }
        .section.active { display: block; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }

        .card {
            background: var(--card); border-radius: var(--radius); padding: 20px;
            margin-bottom: 16px; box-shadow: var(--shadow);
        }
        .card-title { font-size: 1.1em; font-weight: 700; margin-bottom: 16px; display: flex; align-items: center; gap: 8px; }

        .input-group { margin-bottom: 16px; }
        .input-group label { display: block; font-size: 0.85em; font-weight: 600; color: var(--gray); margin-bottom: 6px; }
        .input-field {
            width: 100%; padding: 14px 16px; border: 2px solid #e2e8f0; border-radius: 12px;
            font-size: 1em; font-family: inherit; transition: all 0.2s; background: white;
        }
        .input-field:focus { outline: none; border-color: var(--primary); box-shadow: 0 0 0 4px rgba(118,185,0,0.1); }
        textarea.input-field { min-height: 120px; resize: vertical; }

        .toggle-row { display: flex; align-items: center; justify-content: space-between; padding: 12px 0; border-bottom: 1px solid #f1f5f9; }
        .toggle-row:last-child { border-bottom: none; }
        .toggle-label { font-size: 0.9em; font-weight: 500; }
        .toggle-desc { font-size: 0.75em; color: var(--gray); margin-top: 2px; }
        .toggle-switch {
            position: relative; width: 52px; height: 30px; background: #e2e8f0;
            border-radius: 15px; cursor: pointer; transition: background 0.3s; flex-shrink: 0;
        }
        .toggle-switch.active { background: var(--primary); }
        .toggle-switch::after {
            content: ''; position: absolute; top: 3px; left: 3px; width: 24px; height: 24px;
            background: white; border-radius: 50%; transition: transform 0.3s; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .toggle-switch.active::after { transform: translateX(22px); }

        .btn {
            width: 100%; padding: 16px; border: none; border-radius: 14px;
            font-size: 1em; font-weight: 700; cursor: pointer; transition: all 0.2s;
            display: flex; align-items: center; justify-content: center; gap: 8px;
        }
        .btn-primary { background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%); color: white; }
        .btn-primary:active { transform: scale(0.98); }
        .btn-secondary { background: var(--light); color: var(--dark); }

        .loading-overlay {
            position: fixed; top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(255,255,255,0.9); backdrop-filter: blur(10px);
            display: none; align-items: center; justify-content: center; flex-direction: column;
            z-index: 2000;
        }
        .loading-overlay.active { display: flex; }
        .spinner {
            width: 48px; height: 48px; border: 4px solid #e2e8f0;
            border-top-color: var(--primary); border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        .loading-text { margin-top: 16px; font-weight: 600; color: var(--gray); }

        .result-card {
            background: var(--card); border-radius: var(--radius); padding: 16px;
            margin-bottom: 12px; box-shadow: var(--shadow); border-left: 4px solid var(--primary);
        }
        .result-header { display: flex; gap: 12px; margin-bottom: 12px; }
        .result-icon { width: 40px; height: 40px; background: linear-gradient(135deg, var(--primary), var(--primary-dark));
            border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 1.2em; flex-shrink: 0;
        }
        .result-title { font-weight: 700; font-size: 0.95em; line-height: 1.3; }
        .result-url { font-size: 0.75em; color: var(--gray); margin-top: 2px; word-break: break-all; }
        .result-meta { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 12px; }
        .meta-tag { font-size: 0.7em; padding: 4px 10px; background: var(--light); border-radius: 20px; color: var(--gray); font-weight: 600; }
        .result-content { font-size: 0.85em; line-height: 1.6; color: #334155; }
        .content-preview { max-height: 200px; overflow: hidden; position: relative; }
        .content-preview::after { content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 40px;
            background: linear-gradient(transparent, white);
        }
        .ai-data { background: linear-gradient(135deg, #f0fff4 0%, #dcfce7 100%); border-radius: 12px; padding: 12px; margin-top: 12px; border-left: 4px solid var(--primary); }
        .ai-data h4 { font-size: 0.85em; color: var(--primary-dark); margin-bottom: 8px; }
        .ai-data pre { font-size: 0.75em; overflow-x: auto; white-space: pre-wrap; word-break: break-all; color: #14532d; }

        .toast-container { position: fixed; top: 20px; left: 16px; right: 16px; z-index: 3000; pointer-events: none; }
        .toast {
            background: var(--dark); color: white; padding: 14px 18px; border-radius: 14px;
            margin-bottom: 8px; display: flex; align-items: center; gap: 10px;
            font-size: 0.9em; font-weight: 500; box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            animation: slideIn 0.3s ease; pointer-events: auto;
        }
        .toast.error { background: var(--danger); }
        .toast.success { background: var(--success); }
        @keyframes slideIn { from { transform: translateY(-20px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }

        .empty-state { text-align: center; padding: 40px 20px; color: var(--gray); }
        .empty-state .icon { font-size: 3em; margin-bottom: 12px; opacity: 0.5; }
        .empty-state h3 { font-size: 1.1em; margin-bottom: 4px; color: var(--dark); }
        .empty-state p { font-size: 0.85em; }

        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 2px; }

        @supports (padding-top: env(safe-area-inset-top)) {
            .header { padding-top: calc(20px + env(safe-area-inset-top)); }
        }
    </style>
</head>
<body>
    <div class="app-container">
        <header class="header">
            <div class="header-top">
                <div class="logo">🦅 Claw</div>
                <span class="version">NVIDIA AI</span>
            </div>
            <div class="header-subtitle">Powered by NVIDIA NIM</div>
        </header>

        <div class="status-bar">
            <span class="status-pill online" id="statusApi"><span class="status-dot active"></span>API</span>
            <span class="status-pill warning" id="statusAi"><span class="status-dot"></span>NVIDIA AI</span>
            <span class="status-pill warning" id="statusProxy"><span class="status-dot"></span>Proxies</span>
            <span class="status-pill online" id="statusRedis"><span class="status-dot"></span>Redis</span>
        </div>

        <section class="section active" id="section-scrape">
            <div class="card">
                <div class="card-title">🌐 Scrape URL</div>
                <div class="input-group">
                    <label>Target URL</label>
                    <input type="url" class="input-field" id="scrapeUrl" placeholder="https://example.com/article" value="https://news.ycombinator.com">
                </div>

                <div style="margin-bottom: 16px;">
                    <div class="toggle-row">
                        <div><div class="toggle-label">JavaScript Rendering</div><div class="toggle-desc">For SPAs and dynamic content</div></div>
                        <div class="toggle-switch" data-toggle="js" onclick="toggleSwitch(this)"></div>
                    </div>
                    <div class="toggle-row">
                        <div><div class="toggle-label">AI Extraction</div><div class="toggle-desc">Extract structured data with NVIDIA LLM</div></div>
                        <div class="toggle-switch active" data-toggle="ai" onclick="toggleSwitch(this)"></div>
                    </div>
                    <div class="toggle-row">
                        <div><div class="toggle-label">Auto Schema Detection</div><div class="toggle-desc">Detect page structure automatically</div></div>
                        <div class="toggle-switch active" data-toggle="schema" onclick="toggleSwitch(this)"></div>
                    </div>
                    <div class="toggle-row">
                        <div><div class="toggle-label">Summarize</div><div class="toggle-desc">Generate AI summary</div></div>
                        <div class="toggle-switch" data-toggle="summary" onclick="toggleSwitch(this)"></div>
                    </div>
                    <div class="toggle-row">
                        <div><div class="toggle-label">Use Proxies</div><div class="toggle-desc">Rotate through proxy pool</div></div>
                        <div class="toggle-switch active" data-toggle="proxy" onclick="toggleSwitch(this)"></div>
                    </div>
                </div>

                <div class="input-group" id="aiSchemaGroup">
                    <label>AI Extraction Schema (JSON)</label>
                    <textarea class="input-field" id="aiSchema" rows="4">{"title": "Article title", "author": "Author name", "main_topic": "Main topic", "key_points": "Key points made"}</textarea>
                </div>

                <button class="btn btn-primary" onclick="scrapeUrl()"><span>🚀</span> Start Scraping</button>
            </div>
            <div id="scrapeResults"></div>
        </section>

        <section class="section" id="section-batch">
            <div class="card">
                <div class="card-title">📦 Batch Scrape</div>
                <div class="input-group">
                    <label>URLs (one per line, max 50)</label>
                    <textarea class="input-field" id="batchUrls" rows="6">https://news.ycombinator.com&#10;https://reddit.com/r/programming&#10;https://techcrunch.com</textarea>
                </div>
                <div style="margin-bottom: 16px;">
                    <div class="toggle-row"><div><div class="toggle-label">JavaScript Rendering</div></div><div class="toggle-switch" data-toggle="batchJs" onclick="toggleSwitch(this)"></div></div>
                    <div class="toggle-row"><div><div class="toggle-label">AI Extraction</div></div><div class="toggle-switch" data-toggle="batchAi" onclick="toggleSwitch(this)"></div></div>
                </div>
                <button class="btn btn-primary" onclick="batchScrape()"><span>⚡</span> Batch Process</button>
            </div>
            <div id="batchResults"></div>
        </section>

        <section class="section" id="section-ai">
            <div class="card">
                <div class="card-title">🤖 AI Analysis</div>
                <div class="input-group">
                    <label>Content to Analyze</label>
                    <textarea class="input-field" id="aiContent" rows="6" placeholder="Paste article text here..."></textarea>
                </div>
                <div class="input-group">
                    <label>Analysis Type</label>
                    <select class="input-field" id="aiQueryType" onchange="updateAIForm()">
                        <option value="extract">Extract Structured Data</option>
                        <option value="summarize">Summarize</option>
                        <option value="classify">Classify Content</option>
                        <option value="entities">Extract Entities</option>
                        <option value="qa">Generate Q&A</option>
                    </select>
                </div>
                <div class="input-group" id="extractSchemaGroup">
                    <label>Extraction Schema (JSON)</label>
                    <textarea class="input-field" id="extractSchema" rows="3">{"company": "Company name", "amount": "Funding amount", "investors": "List of investors"}</textarea>
                </div>
                <div class="input-group" id="classifyGroup" style="display:none">
                    <label>Categories (comma-separated)</label>
                    <input type="text" class="input-field" id="classifyCategories" value="news, blog, product, academic, forum">
                </div>
                <div class="input-group" id="qaGroup" style="display:none">
                    <label>Number of Questions</label>
                    <input type="number" class="input-field" id="qaCount" value="5" min="1" max="10">
                </div>
                <button class="btn btn-primary" onclick="analyzeWithAI()"><span>✨</span> Analyze</button>
            </div>
            <div id="aiResults"></div>
        </section>

        <section class="section" id="section-dist">
            <div class="card">
                <div class="card-title">🌐 Distributed Crawl</div>
                <div class="input-group">
                    <label>Start URL</label>
                    <input type="url" class="input-field" id="crawlStartUrl" placeholder="https://example.com" value="https://news.ycombinator.com">
                </div>
                <button class="btn btn-primary" onclick="startCrawl()"><span>🚀</span> Enqueue & Start</button>
            </div>
            <div class="card">
                <div class="card-title">📊 Queue Stats</div>
                <div id="distStats">
                    <div class="empty-state"><div class="icon">📡</div><h3>No Active Crawl</h3><p>Start a crawl to see queue statistics</p></div>
                </div>
                <button class="btn btn-secondary" onclick="refreshStats()" style="margin-top: 12px;"><span>🔄</span> Refresh Stats</button>
            </div>
        </section>

        <nav class="bottom-nav">
            <button class="nav-item active" onclick="showSection('scrape')"><span class="icon">🌐</span><span>Scrape</span></button>
            <button class="nav-item" onclick="showSection('batch')"><span class="icon">📦</span><span>Batch</span></button>
            <button class="nav-item" onclick="showSection('ai')"><span class="icon">🤖</span><span>AI</span></button>
            <button class="nav-item" onclick="showSection('dist')"><span class="icon">🌐</span><span>Distributed</span></button>
        </nav>
    </div>

    <div class="loading-overlay" id="loadingOverlay"><div class="spinner"></div><div class="loading-text">Processing...</div></div>
    <div class="toast-container" id="toastContainer"></div>

    <script>
        const API_BASE = window.location.origin;
        let currentSection = 'scrape';

        function showSection(name) {
            document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
            document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
            document.getElementById('section-' + name).classList.add('active');
            event.currentTarget.classList.add('active');
            currentSection = name;
        }
        function toggleSwitch(el) { el.classList.toggle('active'); }
        function updateAIForm() {
            const type = document.getElementById('aiQueryType').value;
            document.getElementById('extractSchemaGroup').style.display = type === 'extract' ? 'block' : 'none';
            document.getElementById('classifyGroup').style.display = type === 'classify' ? 'block' : 'none';
            document.getElementById('qaGroup').style.display = type === 'qa' ? 'block' : 'none';
        }
        function showLoading(text) { document.querySelector('.loading-text').textContent = text; document.getElementById('loadingOverlay').classList.add('active'); }
        function hideLoading() { document.getElementById('loadingOverlay').classList.remove('active'); }
        function showToast(msg, type) {
            const container = document.getElementById('toastContainer');
            const toast = document.createElement('div');
            toast.className = 'toast ' + type;
            toast.innerHTML = '<span>' + (type === 'success' ? '✅' : type === 'error' ? '❌' : 'ℹ️') + '</span><span>' + msg + '</span>';
            container.appendChild(toast);
            setTimeout(() => toast.remove(), 4000);
        }
        async function apiCall(endpoint, method, body) {
            try {
                const res = await fetch(API_BASE + endpoint, { method: method, headers: { 'Content-Type': 'application/json' }, body: body ? JSON.stringify(body) : undefined });
                if (!res.ok) { const err = await res.json(); throw new Error(err.detail || 'Request failed'); }
                return await res.json();
            } catch (e) { showToast(e.message, 'error'); throw e; }
        }
        function getToggles() { const toggles = {}; document.querySelectorAll('.toggle-switch').forEach(t => { toggles[t.dataset.toggle] = t.classList.contains('active'); }); return toggles; }
        function getSchema() { try { const val = document.getElementById('aiSchema').value; return val ? JSON.parse(val) : null; } catch (e) { return null; } }

        async function scrapeUrl() {
            const url = document.getElementById('scrapeUrl').value;
            if (!url) { showToast('Enter a URL', 'error'); return; }
            showLoading('Scraping with NVIDIA AI...');
            try {
                const toggles = getToggles();
                const result = await apiCall('/api/scrape', 'POST', { url: url, render_js: toggles.js, ai_extract: toggles.ai, ai_schema: getSchema(), detect_schema: toggles.schema, summarize: toggles.summary, proxy_enabled: toggles.proxy });
                displayScrapeResult(result);
                showToast('Scraping complete!', 'success');
            } finally { hideLoading(); }
        }

        function displayScrapeResult(result) {
            const container = document.getElementById('scrapeResults');
            const html = '<div class="result-card"><div class="result-header"><div class="result-icon">📄</div><div><div class="result-title">' + (result.title || 'Untitled') + '</div><div class="result-url">' + result.url + '</div></div></div>' +
                '<div class="result-meta"><span class="meta-tag">' + result.word_count + ' words</span><span class="meta-tag">' + result.links.length + ' links</span><span class="meta-tag">' + result.images.length + ' images</span>' + (result.proxy_used ? '<span class="meta-tag">🔄 ' + result.proxy_used + '</span>' : '') + '</div>' +
                '<div class="result-content"><div class="content-preview">' + (result.content || '').substring(0, 500) + '</div></div>' +
                (result.ai_extracted && Object.keys(result.ai_extracted).length ? '<div class="ai-data"><h4>🤖 NVIDIA AI Extracted</h4><pre>' + JSON.stringify(result.ai_extracted, null, 2) + '</pre></div>' : '') +
                (result.detected_schema && result.detected_schema.content_type ? '<div class="ai-data" style="border-left-color:#f59e0b;background:linear-gradient(135deg,#fffaf0,#fff5eb)"><h4 style="color:#f59e0b">🔍 Schema: ' + result.detected_schema.content_type + '</h4><pre>' + JSON.stringify(result.detected_schema.schema, null, 2) + '</pre></div>' : '') + '</div>';
            container.insertAdjacentHTML('afterbegin', html);
        }

        async function batchScrape() {
            const urls = document.getElementById('batchUrls').value.split('\n').filter(u => u.trim());
            if (!urls.length) { showToast('Enter URLs', 'error'); return; }
            showLoading('Batch scraping ' + urls.length + ' URLs...');
            try {
                const toggles = getToggles();
                const result = await apiCall('/api/batch', 'POST', { urls: urls, render_js: toggles.batchJs, ai_extract: toggles.batchAi });
                displayBatchResult(result);
                showToast('Batch: ' + result.successful + '/' + result.total + ' done', 'success');
            } finally { hideLoading(); }
        }

        function displayBatchResult(result) {
            const container = document.getElementById('batchResults');
            let html = '<div class="result-card"><div class="result-header"><div class="result-icon">📦</div><div><div class="result-title">Batch Results</div><div class="result-url">' + result.successful + ' ok, ' + result.failed + ' failed</div></div></div>';
            result.results.forEach((r, i) => {
                html += '<div style="padding:12px;margin-top:12px;background:' + (r.status === 'success' ? '#f0fff4' : '#fff5f5') + ';border-radius:8px;"><div style="font-weight:600;font-size:0.9em">' + (i+1) + '. ' + (r.title || 'Error') + '</div><div style="font-size:0.8em;color:var(--gray);margin-top:4px">' + r.url + '</div>' + (r.status === 'success' ? '<div class="result-meta" style="margin-top:8px"><span class="meta-tag">' + r.word_count + ' words</span></div>' : '<div style="color:var(--danger);font-size:0.85em;margin-top:8px">⚠️ ' + (r.error || 'Error') + '</div>') + '</div>';
            });
            html += '</div>';
            container.innerHTML = html;
        }

        async function analyzeWithAI() {
            const content = document.getElementById('aiContent').value;
            if (!content.trim()) { showToast('Enter content', 'error'); return; }
            showLoading('Analyzing with NVIDIA AI...');
            try {
                const type = document.getElementById('aiQueryType').value;
                let schema = null, categories = null, numQuestions = 5;
                if (type === 'extract') { try { schema = JSON.parse(document.getElementById('extractSchema').value); } catch(e) { showToast('Invalid JSON', 'error'); return; } }
                else if (type === 'classify') { categories = document.getElementById('classifyCategories').value.split(',').map(c => c.trim()).filter(c => c); }
                else if (type === 'qa') { numQuestions = parseInt(document.getElementById('qaCount').value) || 5; }
                const result = await apiCall('/api/ai/query', 'POST', { content: content, query_type: type, schema: schema, categories: categories, num_questions: numQuestions });
                document.getElementById('aiResults').innerHTML = '<div class="result-card"><div class="result-header"><div class="result-icon">🤖</div><div><div class="result-title">NVIDIA AI: ' + result.query_type + '</div><div class="result-url">Model: ' + (result.model || 'NVIDIA NIM') + '</div></div></div><div class="ai-data" style="margin-top:12px"><pre>' + JSON.stringify(result.result, null, 2) + '</pre></div></div>';
                showToast('AI analysis complete!', 'success');
            } finally { hideLoading(); }
        }

        async function startCrawl() {
            const url = document.getElementById('crawlStartUrl').value;
            if (!url) { showToast('Enter start URL', 'error'); return; }
            showLoading('Enqueueing...');
            try { const result = await apiCall('/api/distributed/enqueue', 'POST', [url]); showToast('Enqueued ' + result.enqueued + ' URL(s)', 'success'); refreshStats(); }
            finally { hideLoading(); }
        }

        async function refreshStats() {
            try {
                const stats = await apiCall('/api/distributed/stats', 'GET');
                const container = document.getElementById('distStats');
                if (!stats.available) { container.innerHTML = '<div class="empty-state"><div class="icon">⚠️</div><h3>Redis Not Connected</h3><p>Configure REDIS_URL environment variable</p></div>'; return; }
                container.innerHTML = '<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px"><div style="background:#f0fff4;padding:16px;border-radius:12px;text-align:center"><div style="font-size:2em;font-weight:700;color:var(--success)">' + stats.pending + '</div><div style="font-size:0.85em;color:var(--gray)">Pending</div></div><div style="background:#e6fffa;padding:16px;border-radius:12px;text-align:center"><div style="font-size:2em;font-weight:700;color:var(--primary)">' + stats.completed + '</div><div style="font-size:0.85em;color:var(--gray)">Completed</div></div><div style="background:#fff5f5;padding:16px;border-radius:12px;text-align:center"><div style="font-size:2em;font-weight:700;color:var(--danger)">' + stats.failed + '</div><div style="font-size:0.85em;color:var(--gray)">Failed</div></div><div style="background:#fffaf0;padding:16px;border-radius:12px;text-align:center"><div style="font-size:2em;font-weight:700;color:var(--warning)">' + stats.visited + '</div><div style="font-size:0.85em;color:var(--gray)">Visited</div></div></div>';
            } catch (e) { console.error(e); }
        }

        async function checkHealth() {
            try {
                const health = await apiCall('/api/health', 'GET');
                document.getElementById('statusApi').innerHTML = '<span class="status-dot active"></span>API';
                document.getElementById('statusApi').className = 'status-pill online';
                const ai = document.getElementById('statusAi');
                const aiConfigured = health.ai && health.ai.configured;
                ai.innerHTML = '<span class="status-dot ' + (aiConfigured ? 'active' : '') + '"></span>NVIDIA AI';
                ai.className = 'status-pill ' + (aiConfigured ? 'online' : 'warning');
                const proxy = document.getElementById('statusProxy');
                proxy.innerHTML = '<span class="status-dot ' + (health.proxies_configured ? 'active' : '') + '"></span>Proxies';
                proxy.className = 'status-pill ' + (health.proxies_configured ? 'online' : 'warning');
                const redis = document.getElementById('statusRedis');
                redis.innerHTML = '<span class="status-dot ' + (health.redis_available ? 'active' : '') + '"></span>Redis';
                redis.className = 'status-pill ' + (health.redis_available ? 'online' : 'warning');
            } catch (e) {
                document.getElementById('statusApi').innerHTML = '<span class="status-dot"></span>API';
                document.getElementById('statusApi').className = 'status-pill offline';
            }
        }

        checkHealth();
        setInterval(checkHealth, 30000);
    </script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(content=HTML_TEMPLATE)

@app.get("/api", response_class=HTMLResponse)
async def api_docs():
    return HTMLResponse(content="""
    <h1>Claw Scraper API v3.1 - NVIDIA Edition</h1>
    <h2>Endpoints:</h2>
    <ul>
    <li><b>GET /api/health</b> - Health check (shows NVIDIA AI status)</li>
    <li><b>POST /api/scrape</b> - Scrape single URL with NVIDIA AI</li>
    <li><b>POST /api/batch</b> - Batch scrape URLs</li>
    <li><b>POST /api/ai/query</b> - AI analysis (extract/summarize/classify/entities/qa)</li>
    <li><b>POST /api/distributed/enqueue</b> - Enqueue URLs</li>
    <li><b>GET /api/distributed/stats</b> - Queue stats</li>
    <li><b>GET /api/proxies/stats</b> - Proxy stats</li>
    </ul>
    <h2>Environment Variables:</h2>
    <ul>
    <li>NVIDIA_API_KEY - Your NVIDIA API key</li>
    <li>NVIDIA_MODEL - Model to use (default: meta/llama-3.1-70b-instruct)</li>
    <li>NVIDIA_BASE_URL - API endpoint (default: https://integrate.api.nvidia.com/v1)</li>
    <li>AI_PROVIDER - "nvidia" or "openai"</li>
    </ul>
    """)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
