from typing import List
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from agents.prompts import REPLY_PROMPT, COMMENT_PROMPT

def load_copilot_agent(model: str = "mistral",
                       max_new_tokens: int = 256,
                       temperature: float = 0.7,
                       top_p: float = 0.9):
    """LLMChain for replying to comments."""
    llm = Ollama(
        model=model,
        temperature=temperature,
        top_p=top_p,
        num_ctx=4096,
        
    )
    return LLMChain(prompt=REPLY_PROMPT, llm=llm)

def load_comment_agent(model: str = "mistral",
                       max_new_tokens: int = 256,
                       temperature: float = 0.7,
                       top_p: float = 0.9):
    """LLMChain for suggesting comments for a post."""
    llm = Ollama(
        model=model,
        temperature=temperature,
        top_p=top_p,
        num_ctx=4096,
        
    )
    return LLMChain(prompt=COMMENT_PROMPT, llm=llm)

def _parse_numbered_list(text: str) -> List[str]:
    """Parse numbered/bulleted list into clean options."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    items, current = [], ""
    for ln in lines:
        if ln[:2].isdigit() or ln.startswith(("1)", "2)", "3)")) or ln[0:2] in ["1.", "2.", "3."]:
            if current:
                items.append(current.strip())
            current = ln
        else:
            current += " " + ln
    if current:
        items.append(current.strip())

    cleaned = []
    for it in items:
        if it[:2] in ["1)", "2)", "3)"]:
            cleaned.append(it[2:].strip())
        elif len(it) > 2 and it[1] == "." and it[0] in "123":
            cleaned.append(it[2:].strip())
        else:
            cleaned.append(it)
    return cleaned[:3] if cleaned else []

def generate_replies(agent_chain: LLMChain, post: str, comment: str,
                     sentiment: str, tone: str, context: str = "") -> List[str]:
    raw = agent_chain.run({
        "post": post,
        "comment": comment,
        "sentiment": sentiment or "neutral",
        "tone": tone,
        "context": context or "N/A"
    })
    parsed = _parse_numbered_list(raw)
    if not parsed:
        parsed = [p.strip("-• ") for p in raw.split("\n") if p.strip()]
        parsed = [p for p in parsed if len(p) > 5][:3]
    return parsed[:3]

def generate_comments(agent_chain: LLMChain, post: str, tone: str, context: str = "") -> List[str]:
    raw = agent_chain.run({
        "post": post,
        "tone": tone,
        "context": context or "N/A"
    })
    parsed = _parse_numbered_list(raw)
    if not parsed:
        parsed = [p.strip("-• ") for p in raw.splitlines() if p.strip()]
        parsed = [p for p in parsed if len(p) > 5][:3]
    return parsed[:3]
