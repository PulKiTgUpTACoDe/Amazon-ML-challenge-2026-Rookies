"""
Abbreviation expansion and advanced text normalization for business entity resolution.
Expands common business abbreviations, legal suffixes, and address shorthand
BEFORE any blocking or feature extraction to maximize string overlap.
"""
import re

# Business legal suffix expansions
BUSINESS_ABBREVIATIONS = {
    # Legal suffixes
    r'\bcorp\b': 'corporation',
    r'\binc\b': 'incorporated',
    r'\bltd\b': 'limited',
    r'\bco\b': 'company',
    r'\bllc\b': 'limited liability company',
    r'\bllp\b': 'limited liability partnership',
    r'\bplc\b': 'public limited company',
    r'\bpvt\b': 'private',
    r'\bpvt\s*ltd\b': 'private limited',
    r'\bintl\b': 'international',
    r'\bnatl\b': 'national',
    r'\bsvcs\b': 'services',
    r'\bsvc\b': 'service',
    r'\bmfg\b': 'manufacturing',
    r'\bassoc\b': 'associates',
    r'\bassn\b': 'association',
    r'\bgrp\b': 'group',
    r'\bhldgs\b': 'holdings',
    r'\bholdings\b': 'holdings',
    r'\bentr\b': 'enterprise',
    r'\benterprises\b': 'enterprises',
    r'\btech\b': 'technology',
    r'\btechnol\b': 'technology',
    r'\bsoln\b': 'solutions',
    r'\bsolns\b': 'solutions',
    r'\bsys\b': 'systems',
    r'\bmgmt\b': 'management',
    r'\bcomm\b': 'communications',
    r'\bdist\b': 'distribution',
    r'\bindus\b': 'industries',
    r'\bpharm\b': 'pharmaceuticals',
    r'\bpharma\b': 'pharmaceuticals',
    r'\blab\b': 'laboratory',
    r'\blabs\b': 'laboratories',
    r'\bfin\b': 'financial',
    r'\bfinl\b': 'financial',
    r'\bconsult\b': 'consulting',
    r'\bconstr\b': 'construction',
    r'\bdba\b': '',  # "doing business as" - remove it
    r'\b&\b': 'and',
}

# Address abbreviation expansions
ADDRESS_ABBREVIATIONS = {
    r'\bst\b': 'street',
    r'\brd\b': 'road',
    r'\bave\b': 'avenue',
    r'\bblvd\b': 'boulevard',
    r'\bdr\b': 'drive',
    r'\bln\b': 'lane',
    r'\bct\b': 'court',
    r'\bpl\b': 'place',
    r'\bpkwy\b': 'parkway',
    r'\bhwy\b': 'highway',
    r'\bfl\b': 'floor',
    r'\bste\b': 'suite',
    r'\bapt\b': 'apartment',
    r'\bbldg\b': 'building',
    r'\bno\b': 'number',
    r'\bn\b': 'north',
    r'\bs\b': 'south',
    r'\be\b': 'east',
    r'\bw\b': 'west',
    r'\bnr\b': 'near',
    r'\bopp\b': 'opposite',
    r'\bdist\b': 'district',
    r'\btaluka\b': 'taluka',
    r'\bvill\b': 'village',
    r'\bnagar\b': 'nagar',
}


def expand_abbreviations(text: str, abbreviation_dict: dict) -> str:
    """Apply all abbreviation expansions to a text string."""
    if not text or not isinstance(text, str):
        return text
    text = text.lower().strip()
    for pattern, replacement in abbreviation_dict.items():
        text = re.sub(pattern, replacement, text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def expand_business_name(name: str) -> str:
    """Expand common business abbreviations in a name."""
    return expand_abbreviations(name, BUSINESS_ABBREVIATIONS)


def expand_address(address: str) -> str:
    """Expand common address abbreviations."""
    return expand_abbreviations(address, ADDRESS_ABBREVIATIONS)
