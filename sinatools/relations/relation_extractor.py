from urllib.request import Request, urlopen
from sinatools.ner.entity_extractor import extract
from sinatools.utils.tokenizer import sentence_tokenizer
from sinatools.utils.entity_utils import distill_entities, sortTags
from .predicate_maps import (
    get_role_category,
    get_arabic_template,
    get_semantic_predicate,
    ROLE_CATEGORIES,
    ARABIC_TEMPLATES,
)
from . import pipe

# ============================ Extract entities and their types ========================
def jsons_to_list_of_lists(json_list):
    return [[d['token'], d['tags']] for d in json_list]

def entities_and_types(sentence):
    output_list = jsons_to_list_of_lists(extract(sentence))
    json_short = distill_entities(output_list)

    entities = {}
    for entity in json_short:
        name = entity[0]
        entity_type = entity[1]
        entities[name] = entity_type

    return entities

# Legacy function for backward compatibility
def get_entity_category(entity_type, categories=None):
    """
    Deprecated: Use get_role_category from predicate_maps instead.
    Kept for backward compatibility with existing code.
    """
    if categories is None:
        categories = ROLE_CATEGORIES
    return get_role_category(entity_type)


def event_argument_relation_extraction(
    document,
    score_threshold=0.50,
    use_semantic_predicates=False,
    fallback_strategy="smart"
):
    """
    Extract event-argument relations from document.

    Args:
        document: Input text document
        score_threshold: Minimum confidence score for relation extraction (default: 0.50)
        use_semantic_predicates: If True, use semantic predicates instead of role categories
        fallback_strategy: Strategy for unmapped predicates ('smart', 'generic', 'role')

    Returns:
        List of extracted relations with TripleID, Subject, Relation, Object, confidence
    """
    sentences = sentence_tokenizer(document)
    output_list = []
    relation = {}
    triple_id = 0

    for sentence in sentences:
        entities = entities_and_types(sentence)
        entity_identifier = {entity: i for entity, i in zip(entities, range(1, len(entities) + 1))}

        event_indices = [i for i, (_, entity_type) in enumerate(entities.items()) if entity_type == 'EVENT']
        arg_event_indices = [i for i, (_, entity_type) in enumerate(entities.items()) if entity_type != 'EVENT']

        for i in event_indices:
            event_entity = list(entities.keys())[i]
            for j in arg_event_indices:
                arg_name = list(entities.keys())[j]
                arg_type = entities[arg_name]

                # Get category for template-based approach (original logic)
                category = get_role_category(arg_type)

                if category and category in ARABIC_TEMPLATES:
                    relation_sentence = f"[CLS] {sentence} [SEP] {event_entity} {ARABIC_TEMPLATES[category]} {arg_name}"
                    predicted_relation = pipe(relation_sentence)
                    score = predicted_relation[0][0]['score']

                    if score > score_threshold:
                        triple_id += 1

                        # Determine predicate based on mode
                        if use_semantic_predicates:
                            # Use semantic predicate mapping
                            predicate = get_semantic_predicate(
                                entities[event_entity],
                                arg_type,
                                fallback_strategy=fallback_strategy
                            )
                        else:
                            # Use original role category
                            predicate = category

                        relation = {
                            "TripleID": triple_id,
                            "Subject": {
                                "ID": entity_identifier[event_entity],
                                "Type": entities[event_entity],
                                "Label": event_entity
                            },
                            "Relation": predicate,
                            "Object": {
                                "ID": entity_identifier[arg_name],
                                "Type": entities[arg_name],
                                "Label": arg_name,
                            },
                            "confidence": f"{score: .2f}"
                        }
                        output_list.append(relation)

    return output_list
