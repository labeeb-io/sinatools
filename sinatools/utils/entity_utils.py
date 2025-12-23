"""Shared utilities for entity processing and IOB tag manipulation.

This module provides common functionality for entity distillation and tag sorting
that is used across NER and relation extraction modules.
"""

from typing import List


def distill_entities(entities: List[List[str]]) -> List[List]:
    """Convert IOB-tagged entities to distilled entity list.

    Takes [token, tags] pairs and returns [token, tag_type, start_pos, end_pos].
    Groups consecutive tokens by IOB tags to merge multi-word entities.

    Args:
        entities: List of [token, tags_string] pairs.
                 tags_string: space-separated IOB tags (e.g., "B-PERS I-PERS")

    Returns:
        List of [token_text, tag_type, start_pos, end_pos] sorted by start_pos.
    """
    # This is list that we put the output what we need
    list_output = []

    # This line go to sort function and save the output to temp_entities
    temp_entities = sortTags(entities)

    # This list help us to make the output,
    temp_list = []

    # initlize the temp_list
    temp_list.append(["", "", 0, 0])
    word_position = 0

    # For each entity, convert ibo to distllir list.
    for entity in temp_entities:
        # This is counter tag of this entity
        counter_tag = 0
        # For each tag
        for tag in str(entity[1]).split():
            # If the counter tag greater than or equal to lenght of templist, if yes then we will append the empty value in templist
            if counter_tag >= len(temp_list):
                temp_list.append(["", "", 0, 0])

            # If tag equal O and word postion of this tag is not equal zero then it will add all
            # not empty eliment of temp list in output list
            if "O" == tag and word_position != 0:
                for j in range(0, len(temp_list)):
                    if temp_list[j][1] != "":
                        list_output.append([temp_list[j][0].strip(), temp_list[j][1], temp_list[j][2], temp_list[j][3]])
                        temp_list[j][0] = ""
                        temp_list[j][1] = ""
                        temp_list[j][2] = word_position
                        temp_list[j][3] = word_position
            # if this tag not equal O, and split by '-' the tag and check the lenght equals two and if the first eliment
            # of the split its B
            elif "O" != tag and len(tag.split("-")) == 2 and tag.split("-")[0] == "B":
                # if the temp_list of counter is not empty then it will append in output list and hten it will
                # initilize by new string and tag in templist of counter
                if temp_list[counter_tag][1] != "":
                    list_output.append([temp_list[counter_tag][0].strip(), temp_list[counter_tag][1], temp_list[counter_tag][2], temp_list[counter_tag][3]])
                temp_list[counter_tag][0] = str(entity[0]) + " "
                temp_list[counter_tag][1] = str(tag).split("-")[1]
                temp_list[counter_tag][2] = word_position
                temp_list[counter_tag][3] = word_position

            # if this tag not equal O, and split by '-' the tag and check the lenght equals two and if the first eliment
            # of the split its O
            elif "O" != tag and len(tag.split("-")) == 2 and tag.split("-")[0] == "I" and word_position != 0:
                # For each of temp_list, check if in this counter tag of templist is same tag with this.tag
                # then will complete if not it will save in output list and cheak another
                for j in range(counter_tag, len(temp_list)):
                    if temp_list[j][1] == tag[2:] and temp_list[j][3] != word_position:
                        temp_list[j][0] += str(entity[0]) + " "
                        temp_list[j][3] += 1
                        break
                    else:
                        if temp_list[j][1] != "":
                            list_output.append([temp_list[j][0].strip(), temp_list[j][1], temp_list[j][2], temp_list[j][3]])
                            temp_list[j][0] = ""
                            temp_list[j][1] = ""
                            temp_list[j][2] = word_position
                            temp_list[j][3] = word_position
            counter_tag += 1
        word_position += 1
    # For each temp_list, at the end of the previous loop, there will be some
    # values in this list, we should save it to the output list
    for j in range(0, len(temp_list)):
        if temp_list[j][1] != "":
            list_output.append([temp_list[j][0].strip(), temp_list[j][1], temp_list[j][2], temp_list[j][3]])
    return sorted(list_output, key=lambda x: (x[2]))


def sortTags(entities: List[List[str]]) -> List[List[str]]:
    """Sort IOB tags per entity, maintaining consistency across sequence.

    Validates and corrects tag sequences (e.g., ensures I- tags follow B- tags).

    Args:
        entities: List of [token, tags_string] pairs.

    Returns:
        Sorted entities with corrected tag sequences.
    """
    temp_entities = entities
    temp_counter = 0
    # For each entity, this loop will sort each tag of entitiy, first it will check if the
    # previous tags has same count of this tag, second will sort the tags and check if this tags is correct
    for entity in temp_entities:
        tags = entity[1].split()
        for tag in tags:
            # if the counter is not 0 then, will complete
            if temp_counter != 0:
                # Check if this tag is equal I-, if yes then it will count how many tag in this tags and
                # count how many tag in previous tags
                if "I-" == tag[0:2]:
                    counter_of_this_tag = 0
                    counter_of_previous_tag = 0
                    for word in tags:
                        if tag.split("-")[1] in word:
                            counter_of_this_tag += 1
                    for word in temp_entities[temp_counter - 1][1].split():
                        if tag.split("-")[1] in word:
                            counter_of_previous_tag += 1
                    # if the counter of previous tag is bigger than counter of this tag, then we
                    # need to add I-tag in this tags
                    if counter_of_previous_tag > counter_of_this_tag:
                        tags.append("I-" + tag.split("-")[1])
        # Sort the tags
        tags.sort()
        # Need to revers the tags because it should begins with I
        tags.reverse()
        # If the counter is not 0 then we can complete
        if temp_counter != 0:
            this_tags = tags
            previous_tags = temp_entities[temp_counter - 1][1].split()
            sorted_tags = []

            # Check if the this tag is not O and previous tags is not O, then will complete,
            # if not then it will ignor this tag
            if "O" not in this_tags and "O" not in previous_tags:
                index = 0
                # For each previous tags, need sort this tag by previous tags if its I, B we can ignor
                for i in previous_tags:
                    j = 0
                    while this_tags and j < len(this_tags):
                        if this_tags[j][0:2] == "I-" and this_tags[j][2:] == i[2:]:
                            sorted_tags.insert(index, this_tags.pop(j))
                            break
                        elif this_tags[j][0:2] == "B-":
                            break
                        j += 1
                    index += 1
            sorted_tags += this_tags
            tags = sorted_tags
        str_tag = " "
        str_tag = str_tag.join(tags)
        str_tag = str_tag.strip()
        temp_entities[temp_counter][1] = str_tag
        temp_counter += 1
    return temp_entities


__all__ = ["distill_entities", "sortTags"]
