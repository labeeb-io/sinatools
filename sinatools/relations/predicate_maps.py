"""
Semantic predicate mapping for relation extraction.

This module provides comprehensive predicate mappings for all entity type pairs,
with dynamic fallback strategies to ensure no type combination is lost.
"""

from typing import Dict, Tuple, Optional, List

# ==============================================================================
# Core Semantic Predicate Map
# ==============================================================================

SEMANTIC_PREDICATE_MAP: Dict[Tuple[str, str], str] = {
    # Person (PERS) relations
    ("PERS", "PERS"): "knowsOf",
    ("PERS", "ORG"): "worksFor",
    ("PERS", "GPE"): "locatedIn",
    ("PERS", "LOC"): "locatedIn",
    ("PERS", "FAC"): "visits",
    ("PERS", "DATE"): "occurredOn",
    ("PERS", "TIME"): "occurredAt",
    ("PERS", "OCC"): "hasOccupation",
    ("PERS", "NORP"): "memberOf",
    ("PERS", "EVENT"): "participatesIn",

    # Organization (ORG) relations
    ("ORG", "PERS"): "employs",
    ("ORG", "ORG"): "relatedTo",
    ("ORG", "GPE"): "locatedIn",
    ("ORG", "LOC"): "locatedIn",
    ("ORG", "FAC"): "occupies",
    ("ORG", "DATE"): "foundedOn",
    ("ORG", "TIME"): "operatesAt",
    ("ORG", "OCC"): "hasFunction",
    ("ORG", "NORP"): "represents",
    ("ORG", "EVENT"): "organizes",

    # Geo-Political Entity (GPE) relations
    ("GPE", "PERS"): "hasResident",
    ("GPE", "ORG"): "hosts",
    ("GPE", "GPE"): "bordersWith",
    ("GPE", "LOC"): "contains",
    ("GPE", "FAC"): "contains",
    ("GPE", "DATE"): "existsOn",
    ("GPE", "TIME"): "existsAt",
    ("GPE", "OCC"): "employsRole",
    ("GPE", "NORP"): "homeTo",
    ("GPE", "EVENT"): "hostsEvent",

    # Location (LOC) relations
    ("LOC", "PERS"): "hasVisitor",
    ("LOC", "ORG"): "contains",
    ("LOC", "GPE"): "partOf",
    ("LOC", "LOC"): "adjacentTo",
    ("LOC", "FAC"): "contains",
    ("LOC", "DATE"): "existsOn",
    ("LOC", "TIME"): "existsAt",
    ("LOC", "OCC"): "hasActivity",
    ("LOC", "NORP"): "homeTo",
    ("LOC", "EVENT"): "hostsEvent",

    # Facility (FAC) relations
    ("FAC", "PERS"): "visited",
    ("FAC", "ORG"): "houses",
    ("FAC", "GPE"): "locatedIn",
    ("FAC", "LOC"): "locatedIn",
    ("FAC", "FAC"): "adjacentTo",
    ("FAC", "DATE"): "builtOn",
    ("FAC", "TIME"): "openAt",
    ("FAC", "OCC"): "hasActivity",
    ("FAC", "NORP"): "serves",
    ("FAC", "EVENT"): "hostsEvent",

    # Date (DATE) relations
    ("DATE", "PERS"): "involves",
    ("DATE", "ORG"): "involves",
    ("DATE", "GPE"): "occurredIn",
    ("DATE", "LOC"): "occurredIn",
    ("DATE", "FAC"): "occurredAt",
    ("DATE", "DATE"): "relatedTo",
    ("DATE", "TIME"): "hasTime",
    ("DATE", "OCC"): "hasActivity",
    ("DATE", "NORP"): "involves",
    ("DATE", "EVENT"): "dateOf",

    # Time (TIME) relations
    ("TIME", "PERS"): "involves",
    ("TIME", "ORG"): "involves",
    ("TIME", "GPE"): "occurredIn",
    ("TIME", "LOC"): "occurredIn",
    ("TIME", "FAC"): "occurredAt",
    ("TIME", "DATE"): "partOf",
    ("TIME", "TIME"): "relatedTo",
    ("TIME", "OCC"): "hasActivity",
    ("TIME", "NORP"): "involves",
    ("TIME", "EVENT"): "timeOf",

    # Occupation (OCC) relations
    ("OCC", "PERS"): "performedBy",
    ("OCC", "ORG"): "performedBy",
    ("OCC", "GPE"): "performedIn",
    ("OCC", "LOC"): "performedIn",
    ("OCC", "FAC"): "performedAt",
    ("OCC", "DATE"): "performedOn",
    ("OCC", "TIME"): "performedAt",
    ("OCC", "OCC"): "relatedTo",
    ("OCC", "NORP"): "associatedWith",
    ("OCC", "EVENT"): "roleIn",

    # Nationality/Group (NORP) relations
    ("NORP", "PERS"): "includesMember",
    ("NORP", "ORG"): "affiliatedWith",
    ("NORP", "GPE"): "originatesFrom",
    ("NORP", "LOC"): "originatesFrom",
    ("NORP", "FAC"): "uses",
    ("NORP", "DATE"): "existsOn",
    ("NORP", "TIME"): "existsAt",
    ("NORP", "OCC"): "hasRole",
    ("NORP", "NORP"): "relatedTo",
    ("NORP", "EVENT"): "participatesIn",

    # Event (EVENT) relations
    ("EVENT", "PERS"): "involves",
    ("EVENT", "ORG"): "involves",
    ("EVENT", "GPE"): "occurredIn",
    ("EVENT", "LOC"): "occurredAt",
    ("EVENT", "FAC"): "occurredAt",
    ("EVENT", "DATE"): "occurredOn",
    ("EVENT", "TIME"): "occurredAt",
    ("EVENT", "OCC"): "involves",
    ("EVENT", "NORP"): "affects",
    ("EVENT", "EVENT"): "relatedTo",
}

# ==============================================================================
# Role-Based Category Mappings (from original package)
# ==============================================================================

ROLE_CATEGORIES: Dict[str, List[str]] = {
    'agent': ['PERS', 'NORP', 'OCC', 'ORG'],
    'location': ['LOC', 'FAC', 'GPE'],
    'happened_at': ['DATE', 'TIME'],
    'event': ['EVENT'],
}

# Arabic templates for role-based predicates (used in original Hadath model)
ARABIC_TEMPLATES: Dict[str, str] = {
    'location': 'مكان حدوث',
    'agent': 'أحد المتأثرين في',
    'happened_at': 'تاريخ حدوث',
    'event': 'الحدث',
}

# ==============================================================================
# Predicate Priority Tiers (for intelligent fallback)
# ==============================================================================

# Tier 1: High-confidence semantic predicates
TIER1_PREDICATES = {
    ("PERS", "ORG"): "worksFor",
    ("ORG", "PERS"): "employs",
    ("PERS", "GPE"): "locatedIn",
    ("EVENT", "GPE"): "occurredIn",
    ("EVENT", "DATE"): "occurredOn",
    ("EVENT", "PERS"): "involves",
}

# Tier 2: Common predicates with moderate confidence
TIER2_PREDICATES = {
    ("ORG", "GPE"): "locatedIn",
    ("ORG", "DATE"): "foundedOn",
    ("GPE", "PERS"): "hasResident",
    ("GPE", "ORG"): "hosts",
}

# ==============================================================================
# Dynamic Predicate Generation Strategies
# ==============================================================================

def get_semantic_predicate(
    subject_type: str,
    object_type: str,
    fallback_strategy: str = "smart"
) -> str:
    """
    Get semantic predicate for entity type pair with intelligent fallback.

    Args:
        subject_type: Entity type of subject (e.g., 'PERS', 'ORG')
        object_type: Entity type of object (e.g., 'GPE', 'DATE')
        fallback_strategy: Strategy for unmapped pairs
            - 'smart': Use semantic rules based on entity classes
            - 'generic': Use simple concatenation
            - 'role': Use role-based mapping

    Returns:
        Semantic predicate string

    Examples:
        >>> get_semantic_predicate('PERS', 'ORG')
        'worksFor'
        >>> get_semantic_predicate('CUSTOM', 'GPE')  # Unknown type
        'associatedWith'
    """
    key = (subject_type, object_type)

    # Try exact match in main map
    if key in SEMANTIC_PREDICATE_MAP:
        return SEMANTIC_PREDICATE_MAP[key]

    # Try tier 1 high-confidence predicates
    if key in TIER1_PREDICATES:
        return TIER1_PREDICATES[key]

    # Try tier 2 common predicates
    if key in TIER2_PREDICATES:
        return TIER2_PREDICATES[key]

    # Apply fallback strategy
    if fallback_strategy == "smart":
        return _smart_predicate_fallback(subject_type, object_type)
    elif fallback_strategy == "role":
        return _role_based_predicate(subject_type, object_type)
    else:
        return _generic_predicate_fallback(subject_type, object_type)


def _smart_predicate_fallback(subject_type: str, object_type: str) -> str:
    """
    Smart fallback using semantic rules for unknown type combinations.

    Rules:
    - AGENT types (PERS, ORG) + LOCATION types → 'locatedIn'
    - AGENT types + TEMPORAL types → 'occurredOn'
    - LOCATION types + AGENT types → 'contains'
    - EVENT + any type → 'involves' or 'occurredIn/On/At'
    - Same type → 'relatedTo'
    - Default → 'associatedWith'
    """
    subj_upper = subject_type.upper()
    obj_upper = object_type.upper()

    # Same type relations
    if subj_upper == obj_upper:
        return "relatedTo"

    # Agent + Location
    if subj_upper in ("PERS", "ORG", "NORP") and obj_upper in ("GPE", "LOC", "FAC"):
        return "locatedIn"

    # Location + Agent
    if subj_upper in ("GPE", "LOC", "FAC") and obj_upper in ("PERS", "ORG", "NORP"):
        return "contains"

    # Agent + Temporal
    if subj_upper in ("PERS", "ORG", "NORP", "EVENT") and obj_upper in ("DATE", "TIME"):
        return "occurredOn"

    # Temporal + Agent/Location
    if subj_upper in ("DATE", "TIME") and obj_upper in ("PERS", "ORG", "GPE", "LOC"):
        return "involves"

    # Event relations
    if subj_upper == "EVENT":
        if obj_upper in ("GPE", "LOC", "FAC"):
            return "occurredIn"
        elif obj_upper in ("DATE", "TIME"):
            return "occurredOn"
        else:
            return "involves"

    # Default: generic association
    return "associatedWith"


def _role_based_predicate(subject_type: str, object_type: str) -> str:
    """
    Role-based predicate mapping (compatible with original Hadath templates).

    Returns predicates aligned with the original role categories:
    - agent, location, happened_at
    """
    subj_role = get_role_category(subject_type)
    obj_role = get_role_category(object_type)

    if subj_role == 'event' and obj_role:
        return obj_role  # Return the role directly for EVENT relations

    if obj_role == 'location':
        return 'locatedIn'
    elif obj_role == 'happened_at':
        return 'occurredOn'
    elif obj_role == 'agent':
        return 'involves'

    return _generic_predicate_fallback(subject_type, object_type)


def _generic_predicate_fallback(subject_type: str, object_type: str) -> str:
    """
    Generic fallback: creates readable predicate from type names.

    Ensures NO type combination is ever lost - always returns a valid predicate.
    """
    subj_clean = subject_type.replace("_", "").replace("-", "")
    obj_clean = object_type.replace("_", "").replace("-", "")
    return f"{subj_clean}_relatedTo_{obj_clean}"


# ==============================================================================
# Role Category Mapping
# ==============================================================================

def get_role_category(entity_type: str) -> Optional[str]:
    """
    Map entity type to role category.

    Args:
        entity_type: Entity type string (e.g., 'PERS', 'GPE')

    Returns:
        Role category ('agent', 'location', 'happened_at', 'event') or None

    Examples:
        >>> get_role_category('PERS')
        'agent'
        >>> get_role_category('GPE')
        'location'
        >>> get_role_category('UNKNOWN')
        None
    """
    entity_upper = entity_type.upper()
    for category, types in ROLE_CATEGORIES.items():
        if entity_upper in types:
            return category
    return None


def get_arabic_template(category: str) -> Optional[str]:
    """
    Get Arabic template for a role category.

    Args:
        category: Role category ('agent', 'location', 'happened_at')

    Returns:
        Arabic template string or None
    """
    return ARABIC_TEMPLATES.get(category)


# ==============================================================================
# Predicate Validation & Statistics
# ==============================================================================

def get_all_mapped_types() -> List[str]:
    """Get list of all entity types that have explicit mappings."""
    types_set = set()
    for (subj, obj) in SEMANTIC_PREDICATE_MAP.keys():
        types_set.add(subj)
        types_set.add(obj)
    return sorted(types_set)


def get_coverage_stats() -> Dict[str, int]:
    """
    Get statistics about predicate mapping coverage.

    Returns:
        Dictionary with coverage statistics
    """
    mapped_types = get_all_mapped_types()
    total_combinations = len(mapped_types) ** 2
    mapped_combinations = len(SEMANTIC_PREDICATE_MAP)

    return {
        'mapped_types': len(mapped_types),
        'total_combinations': total_combinations,
        'mapped_combinations': mapped_combinations,
        'coverage_percentage': round((mapped_combinations / total_combinations) * 100, 2),
    }


def validate_predicate_map() -> bool:
    """
    Validate predicate map for consistency.

    Returns:
        True if map is valid, False otherwise
    """
    for (subj, obj), predicate in SEMANTIC_PREDICATE_MAP.items():
        if not isinstance(subj, str) or not isinstance(obj, str):
            return False
        if not isinstance(predicate, str) or not predicate:
            return False
        if not predicate.replace("_", "").isalnum():
            return False
    return True


# ==============================================================================
# Exports
# ==============================================================================

__all__ = [
    'SEMANTIC_PREDICATE_MAP',
    'ROLE_CATEGORIES',
    'ARABIC_TEMPLATES',
    'get_semantic_predicate',
    'get_role_category',
    'get_arabic_template',
    'get_all_mapped_types',
    'get_coverage_stats',
    'validate_predicate_map',
]
