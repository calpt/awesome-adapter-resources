import os
from dataclasses import dataclass
import re
from typing import List, Union

import requests
import slugify
from ruamel.yaml import YAML


TEMPLATE = os.path.join(os.path.dirname(__file__), "TEMPLATE.md")


yaml = YAML()


@dataclass
class Section:
    index: int
    title: str
    has_subsections: bool
    items: Union[List["Section"], List["Item"]]


@dataclass
class Item:
    title: str
    description: str
    authors: List[str]
    venue: str
    year: str
    citations: int
    pdf_url: str
    code_url: str
    website_url: str
    semantic_scholar_url: str
    tags: List[str]


def get_semanticscholar_data(items_data: List[dict]) -> List[dict]:
    idxs, paper_ids = [], []
    for idx, item_data in enumerate(items_data):
        if paper_data := item_data.get("paper", None):
            idxs.append(idx)
            if semantic_scholar_id := paper_data.get("semantic_scholar_id", None):
                paper_ids.append(semantic_scholar_id)
            elif acl_id := paper_data.get("acl_id", None):
                paper_ids.append(f"ACL:{acl_id}")
            elif arxiv_id := paper_data.get("arxiv_id", None):
                paper_ids.append(f"ARXIV:{arxiv_id}")
            else:
                raise ValueError(f"Invalid paper data: {paper_data}")
    if len(paper_ids) > 0:
        r = requests.post(
            "https://api.semanticscholar.org/graph/v1/paper/batch",
            params={"fields": "title,abstract,tldr,authors,venue,year,citationCount,url"},
            json={"ids": paper_ids},
        )
        r.raise_for_status()
        ss_data = r.json()
        for idx, ss_item_data in zip(idxs, ss_data):
            ss_item_data["authors"] = [author["name"] for author in ss_item_data["authors"]]
            items_data[idx]["paper"].update(ss_item_data)

    return items_data


def build_pdf_url(paper_data: dict) -> str:
    if pdf_url := paper_data.get("pdf", None):
        return pdf_url
    elif acl_id := paper_data.get("acl_id", None):
        return f"https://aclanthology.org/{acl_id}.pdf"
    elif arxiv_id := paper_data.get("arxiv_id", None):
        return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    else:
        raise ValueError(f"Unable to find valid PDF: {paper_data}")


def load_item(item_data: dict) -> Item:
    paper_data = item_data.get("paper", {})
    return Item(
        title=item_data.get("title", None),
        description=paper_data.get("tldr", {}).get("text", None),
        authors=paper_data.get("authors", []),
        venue=paper_data.get("venue", None),
        year=paper_data.get("year", None),
        citations=int(paper_data.get("citationCount", -1)),
        pdf_url=build_pdf_url(paper_data) if paper_data else None,
        code_url=item_data.get("code", None),
        website_url=item_data.get("website", None),
        semantic_scholar_url=paper_data.get("url", None),
        tags=item_data.get("tags", []),
    )


def read_section(section_data: dict, index=0) -> Section:
    items_data = get_semanticscholar_data(section_data.get("items", []))
    return Section(
        index=section_data.get("index", index),
        title=section_data["title"],
        has_subsections=False,
        items=[load_item(item_data) for item_data in items_data],
    )


def read_data_file(file: str) -> Section:
    print(f"Reading {file}...")
    with open(file) as f:
        data = yaml.load(f)
    if "sections" in data:
        return Section(
            index=data["index"],
            title=data["title"],
            has_subsections=True,
            items=[read_section(section_data, i) for i, section_data in enumerate(data["sections"])],
        )
    else:
        return read_section(data)


def read_data(folder: str) -> List[Section]:
    sections = []
    for file in os.listdir(folder):
        sections.append(read_data_file(os.path.join(folder, file)))

    return sections


def count_items(sections: List[Section]) -> int:
    count = 0
    for section in sections:
        if section.has_subsections:
            count += count_items(section.items)
        else:
            count += len(section.items)

    return count


def _enc_tag(tag: str) -> str:
    return tag.replace(" ", "%20").replace("-", "--")


def _enc_link(link: str) -> str:
    return slugify.slugify(link)


def generate_toc(sections: List[Section]) -> str:
    toc = []
    for section in sections:
        if section.has_subsections:
            toc.append(f"- [{section.title}](#{_enc_link(section.title)})")
            for subsection in section.items:
                toc.append(f"  - [{subsection.title}](#{_enc_link(subsection.title)})")
        else:
            toc.append(f"- [{section.title}](#{_enc_link(section.title)})")

    return "\n".join(toc)


def generate_item_md(item: Item) -> str:
    item_md = []
    # Title & Tags
    tags = " ".join(f"![](https://img.shields.io/badge/-{_enc_tag(tag)}-blue)" for tag in item.tags)
    # Add paper citations
    # if item.citations > -1:
    #     tags = f"![Citations](https://img.shields.io/badge/citations-{item.citations}-violet?logo=Semantic%20Scholar) " + tags
    # Add GH stars
    if item.code_url and (gh := re.search("github.com/([^/]+)/([^/]+)", item.code_url)):
        tags = (
            f"![GitHub Repo stars](https://img.shields.io/github/stars/{gh.group(1)}/{gh.group(2)}?color=yellow&logo=github) "
            + tags
        )
    item_md.append(f"- **{item.title}**&nbsp; {tags}")
    # Venue
    if item.venue:
        item_md.append(item.venue)
    # Authors & Year
    authors = "_" + ", ".join(item.authors) + "_" if item.authors else ""
    year = f"({item.year})" if item.year else ""
    item_md.append(f"{authors} {year}")
    # Description
    if item.description:
        description = f"""<details>
    <summary>TLDR</summary>
    {item.description}
  </details>"""
    else:
        description = ""
    item_md.append(description)
    # Links
    links = []
    if item.pdf_url:
        links.append(f"[[Paper PDF]]({item.pdf_url})")
    if item.code_url:
        links.append(f"[[Code]]({item.code_url})")
    if item.website_url:
        links.append(f"[[Website]]({item.website_url})")
    if item.semantic_scholar_url:
        links.append(f"[[Semantic Scholar]]({item.semantic_scholar_url})")
    if len(links) > 0:
        item_md.append("&nbsp; ".join(links))

    return "\n\n  ".join(item_md) + "\n\n"


def generate_section_md(section: Section, level: int = 2) -> str:
    md = f"{'#' * level} {section.title}\n\n"
    if section.has_subsections:
        for subsection in section.items:
            md += generate_section_md(subsection, level + 1)
    else:
        for item in section.items:
            md += generate_item_md(item)

    return md


if __name__ == "__main__":
    DATA_DIR = "data"
    OUTPUT_FILE = "README.md"

    sections = sorted(read_data(DATA_DIR), key=lambda s: s.index)
    count = count_items(sections)
    toc = generate_toc(sections)
    section_mds = [generate_section_md(section) for section in sections]
    with open(TEMPLATE) as f:
        template = f.read()
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(template.format(count, toc, "\n".join(section_mds)))
