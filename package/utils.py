import yaml
import re
from pathlib import Path

def load_keywords(config_path: str | Path = "keywords.config.yaml") -> dict:
    """Load keywords configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file.
        
    Returns:
        Dictionary containing the parsed YAML configuration.
        
    Raises:
        FileNotFoundError: If the configuration file does not exist.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def contains_keyword(text: str, keywords: list[str]) -> bool:
    """Check if any keyword exists in text using exact word boundary matching.
    
    Args:
        text: The text to search in.
        keywords: List of keywords to search for.
        
    Returns:
        True if any keyword is found, False otherwise.
    """
    pattern = r'\b(' + '|'.join(re.escape(k) for k in keywords) + r')\b'
    return bool(re.search(pattern, text, re.IGNORECASE))

def contains_keyword_fuzzy(text: str, keywords: list[str], threshold: int = 85) -> bool:
    """Check if any keyword exists in text using fuzzy matching to handle misspellings.
    
    Args:
        text: The text to search in.
        keywords: List of keywords to search for.
        threshold: Similarity threshold (0-100). Higher values require closer matches.
            Default is 85.
            
    Returns:
        True if any keyword matches above the threshold, False otherwise.
    """
    from rapidfuzz import fuzz
    text_lower = text.lower()
    for keyword in keywords:
        if fuzz.partial_ratio(keyword.lower(), text_lower) >= threshold:
            return True
    return False
