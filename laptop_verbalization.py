# Run the code for the ontology you want to verbalize
from owlready2 import get_ontology
import json

# LAPTOP ONTOLOGY VERBALIZATION

# Load the ontology
onto = get_ontology("LAPTOP_ONTOLOGY_PATH").load()

# TYPE 1 SENTIMENT EXPRESSIONS

# Function to get synonyms (labels) for a class, excluding those with "Mention"
def get_synonyms(cls):
    synonyms = []
    # Get rdfs:label
    if cls.label:
        synonyms.extend(str(label) for label in cls.label if "Mention" not in str(label))
    # Search for skos:prefLabel and skos:altLabel properties
    for prop in onto.search(iri="*prefLabel") + onto.search(iri="*altLabel"):
        values = prop[cls]
        synonyms.extend(str(v) for v in values if str(v) not in synonyms and "Mention" not in str(v))
    # Search for lex properties
    for prop in onto.search(iri="*lex"):
        values = prop[cls]
        synonyms.extend(str(v) for v in values if str(v) not in synonyms and "Mention" not in str(v))
    return synonyms if synonyms else [cls.name] if "Mention" not in cls.name else []

# Function to get all relevant subclasses recursively, excluding those with "Property", "Action", or "Sentiment"
def get_relevant_subclasses(cls):
    relevant = []
    for subclass in cls.subclasses():
        # Skip if subclass has "Property", "Action", or "Sentiment" in its name
        if "Property" in subclass.name or "Action" in subclass.name or "Sentiment" in subclass.name:
            continue
        relevant.append(subclass)
        # Recursively get subclasses of this subclass
        relevant.extend(get_relevant_subclasses(subclass))
    return relevant

# Function to verbalize type 1 sentiment expressions
def verbalize_type1_sentiments_laptop():
    # Find key classes containing type 1 expressions
    sentiment = onto.search_one(iri="*SentimentValue")
    positive = onto.search_one(iri="*Positive")
    negative = onto.search_one(iri="*Negative")
    neutral = onto.search_one(iri="*Neutral")
    generic_positive = onto.search_one(iri="*GenericPositiveSentiment")
    generic_negative = onto.search_one(iri="*GenericNegativeSentiment")
    generic_neutral = onto.search_one(iri="*GenericNeutralSentiment")
    entity_mention = onto.search_one(iri="*AspectMention")

    # Check if the required classes are found
    if not all([sentiment, positive, negative, generic_positive, generic_negative, entity_mention]):
        missing = []
        if not sentiment:
            missing.append("Sentiment/SentimentValue")
        if not positive:
            missing.append("Positive")
        if not negative:
            missing.append("Negative")
        if not generic_positive:
            missing.append("GenericPositiveSentiment")
        if not generic_negative:
            missing.append("GenericNegativeSentiment")
        if not entity_mention:
            missing.append("EntityMention")
        print(f"Error: Missing classes: {', '.join(missing)}")
        print("\nAll classes in ontology for debugging:")
        for cls in onto.classes():
            print(f"- {cls.name} (IRI: {cls.iri})")
        return

    # Check if hierarchy is correct
    if not (generic_positive.is_a and positive in generic_positive.is_a):
        print("Warning: GenericPositiveSentiment is not a subclass of Positive.")
    if not (generic_negative.is_a and negative in generic_negative.is_a):
        print("Warning: GenericNegativeSentiment is not a subclass of Negative.")
    if not (positive.is_a and sentiment in positive.is_a) or not (negative.is_a and sentiment in negative.is_a):
        print("Warning: Positive or Negative is not a subclass of Sentiment/SentimentValue.")

    # Find aspect classes (subclasses of EntityMention)
    aspect_classes = list(entity_mention.subclasses())
    if not aspect_classes:
        print("Error: No subclasses of EntityMention found.")
        print("\nAll classes in ontology for debugging:")
        for cls in onto.classes():
            print(f"- {cls.name} (IRI: {cls.iri})")
        return

    # List to store verbalizations in
    verbalizations1L = []

    # Go through subclasses of Sentiment
    for cls in sentiment.descendants():
        # Skip the superclass itself (only focus on its elements/subclasses)
        if cls in [sentiment, positive, negative, neutral, generic_positive, generic_negative, generic_neutral]:
            continue

        # Get superclasses
        superclasses = cls.is_a
        is_type1 = False
        sentiment_value = None

        # Check for type 1
        if generic_positive in superclasses or any(sc == generic_positive for sc in superclasses):
            is_type1 = True
            sentiment_value = "Positive"
        elif generic_negative in superclasses or any(sc == generic_negative for sc in superclasses):
            is_type1 = True
            sentiment_value = "Negative"
        elif generic_neutral in superclasses or any(sc == generic_neutral for sc in superclasses):
            is_type1 = True
            sentiment_value = "Neutral"

        if is_type1:
            # Get synonyms for the sentiment class, excluding "Mention"
            sentiment_synonyms = get_synonyms(cls)
            if not sentiment_synonyms:
                continue  # Skips this step if no valid synonyms without "Mention" are found

            # Verbalize for each aspect and its synonyms (by definiton of type 1)
            for aspect_cls in aspect_classes:
                # Skip if the aspect is a sentiment-related class
                if "Sentiment" in aspect_cls.name or aspect_cls == sentiment:
                    continue
                # Get all relevant subclasses recursively (excluding "Property"", "Action", and "Sentiment")
                aspect_elements = get_relevant_subclasses(aspect_cls)
                if not aspect_elements:
                    continue  # Skip if no relevant subclasses
                # Get synonyms for each element (subclass), excluding "Mention"
                for element in aspect_elements:
                    aspect_synonyms = get_synonyms(element)
                    for aspect_name in aspect_synonyms:
                        # Generate verbalization for each sentiment synonym
                        for sentiment_name in sentiment_synonyms:
                            verbalization1L = f"{sentiment_name} {aspect_name} is {sentiment_value}."
                            verbalizations1L.append(verbalization1L)

    # if verbalizations1L:
    #     return sorted(set(verbalizations1L))
    # else:
    #     return []

    # Return the sorted unique verbalizations
    return sorted(set(verbalizations1L)) if verbalizations1L else []

# Assuming verbalize_type2_sentiments_laptop and verbalize_type3_sentiments are defined elsewhere
type1_verbalizations = verbalize_type1_sentiments_laptop()


# Use this code to print the type 1 verbalizations to check
#     # Print the verbalizations
#     if verbalizations1L:
#         print("")
#         for v in sorted(set(verbalizations1L)):  # Remove duplicates
#             print(v)
#     else:
#         print("\nNo Type-1 Sentiment Verbalizations or Entity aspects found (excluding 'Mention').")
#         # Debug: List all EntityMention subclasses and their subclasses
#         print("\nDebugging: EntityMention subclasses and their elements:")
#         for aspect_cls in aspect_classes:
#             elements = list(aspect_cls.subclasses())
#             print(f"- {aspect_cls.name}: {elements if elements else 'No subclasses'}")

# verbalize_type1_sentiments_laptop()

    # Return the verbalizations as a sorted, unique list
    return sorted(set(verbalizations1L))


# TYPE 2 SENTIMENT EXPRESSIONS

# Function to get synonyms (labels) for a class, excluding those with "Mention"
def get_synonyms(cls):
    synonyms = []
    # Get rdfs:label
    if cls.label:
        synonyms.extend(str(label) for label in cls.label if "Mention" not in str(label))
    # Search for skos:prefLabel and skos:altLabel properties
    for prop in onto.search(iri="*prefLabel") + onto.search(iri="*altLabel"):
        values = prop[cls]
        synonyms.extend(str(v) for v in values if str(v) not in synonyms and "Mention" not in str(v))
    # Search for lex properties
    for prop in onto.search(iri="*lex"):
        values = prop[cls]
        synonyms.extend(str(v) for v in values if str(v) not in synonyms and "Mention" not in str(v))
    return synonyms if synonyms else [cls.name] if "Mention" not in cls.name else []

# Function to get all relevant subclasses recursively, excluding those with "Property", "Action", or "Mention"
def get_relevant_subclasses(cls):
    relevant = []
    for subclass in cls.subclasses():
        # Skip if subclass has "Property" or "Action" in its name
        if "Property" in subclass.name or "Action" in subclass.name or "Sentiment" in subclass.name or "Os" in subclass.name:
            continue
        relevant.append(subclass)
        # Recursively get subclasses of this subclass
        relevant.extend(get_relevant_subclasses(subclass))
    return relevant

# Function to verbalize type 2 sentiment expressions using elements within sentiment classes
def verbalize_type2_sentiments_laptop():
    # Find the superclasses used for this type
    sentiment = onto.search_one(iri="*SentimentValue")
    positive = onto.search_one(iri="*Positive")
    negative = onto.search_one(iri="*Negative")
    entity_mention = onto.search_one(iri="*AspectMention")
    # programs_mention = onto.search_one(iri="*ProgramsMention")

    # Type 2 sentiment classes (subclasses of Positive and Negative)
    controls_positive = onto.search_one(iri="*ControlsPositiveSentiment")
    design_positive = onto.search_one(iri="*Design_featuresPositiveSentiment")
    design_negative = onto.search_one(iri="*Design_featuresNegativeSentiment")
    display_positive = onto.search_one(iri="*DisplayPositiveSentiment")
    display_negative = onto.search_one(iri="*DisplayNegativeSentiment")
    fans_positive = onto.search_one(iri="*Fans_coolingPositiveSentiment")
    fans_negative = onto.search_one(iri="*Fans_coolingNegativeSentiment")
    price_positive = onto.search_one(iri="*PricePositiveSentiment")
    price_negative = onto.search_one(iri="*PriceNegativeSentiment")
    mouse_positive = onto.search_one(iri="*MousePositiveSentiment")
    software_negative = onto.search_one(iri="*SofwareNegativeSentiment")

    # Type 2 aspect classes (subclasses of EntityMention)
    controls_mention = onto.search_one(iri="*ControlsMention")
    design_mention = onto.search_one(iri="*Design_featuresMention")
    display_mention = onto.search_one(iri="*DisplayMention")
    fans_mention = onto.search_one(iri="*Fans_coolingMention")
    price_mention = onto.search_one(iri="*PriceMention")
    mouse_mention = onto.search_one(iri="*MouseMention")
    software_mention = onto.search_one(iri="*ProgramsMention")

    # Check if required classes are found
    required_classes = {
        "SentimentValue": sentiment,
        "Positive": positive,
        "Negative": negative,
        "ControlsPositiveSentiment": controls_positive,
        "Design_featuresPositiveSentiment": design_positive,
        "Design_featuresNegativeSentiment": design_negative,
        "DisplayPositiveSentiment": display_positive,
        "DisplayNegativeSentiment": display_negative,
        "Fans_coolingPositiveSentiment": fans_positive,
        "Fans_coolingNegativeSentiment": fans_negative,
        "PricePositiveSentiment": price_positive,
        "PriceNegativeSentiment": price_negative,
        "MousePositiveSentiment": mouse_positive,
        "SofwareNegativeSentiment": software_negative,
        "AspectMention": entity_mention,
        "ControlsMention": controls_mention,
        "PriceMention": price_mention,
        "Design_featuresMention": design_mention,
        "DisplayMention": display_mention,
        "Fans_coolingMention": fans_mention,
        "MouseMention": mouse_mention,
        "ProgramsMention": software_mention
    }
    missing = [name for name, cls in required_classes.items() if not cls]
    if missing:
        print(f"Error: Missing classes: {', '.join(missing)}")
        print("\nAll classes in ontology for debugging:")
        for cls in onto.classes():
            print(f"- {cls.name} (IRI: {cls.iri})")
        return

    # Check if hierarchy for sentiment classes is correct
    for cls, name in [
        (price_positive, "PricePositiveSentiment"),
        (price_negative, "PriceNegativeSentiment"),
        (controls_positive, "ControlsPositiveSentiment"),
        (design_positive, "Design_featuresPositiveSentiment"),
        (design_negative, "Design_featuresNegativeSentiment"),
        (display_positive, "DisplayPositiveSentiment"),
        (display_negative, "DisplayNegativeSentiment"),
        (fans_positive, "Fans_coolingPositiveSentiment"),
        (fans_negative, "Fans_coolingNegativeSentiment"),
        (mouse_positive, "MousePositiveSentiment"),
        (software_negative, "SoftwareNegativeSentiment")

    ]:
        expected_super = positive if "Positive" in name else negative
        if not (cls and cls.is_a and expected_super in cls.is_a):
            print(f"Warning: {name} is not a subclass of {expected_super.name}.")

    # Connect sentiment classes to their corresponding aspect classes (according to definition type 2)
    sentiment_to_aspect = {
        price_positive: price_mention,
        price_negative: price_mention,
        controls_positive: controls_mention,
        design_positive: design_mention,
        design_negative: design_mention,
        display_positive: display_mention,
        display_negative: display_mention,
        fans_positive: fans_mention,
        fans_negative: fans_mention,
        mouse_positive: mouse_mention,
        software_negative: software_mention
    }

    # List to store the verbalizations
    verbalizations2L = []

    # Go through type 2 sentiment classes to get the elements in the class
    for sentiment_cls, aspect_mention_cls in sentiment_to_aspect.items():
        if not sentiment_cls or not aspect_mention_cls:
            continue  # Skip if either class is missing

        # Determine sentiment value
        sentiment_value = "Positive" if "Positive" in sentiment_cls.name else "Negative"

        # Get all relevant subclasses of the sentiment class
        sentiment_elements = get_relevant_subclasses(sentiment_cls)
        if not sentiment_elements:
            print(f"Warning: No relevant subclasses found for {sentiment_cls.name}.")
            continue

        # Get all relevant subclasses of the aspect class (excluding "Property", "Action", and "Mention")
        aspect_elements = get_relevant_subclasses(aspect_mention_cls)
        if not aspect_elements:
            print(f"Warning: No relevant subclasses found for {aspect_mention_cls.name}.")
            continue

        # Verbalize for each sentiment element and aspect element
        for sentiment_element in sentiment_elements:
            sentiment_synonyms = get_synonyms(sentiment_element)
            for aspect_element in aspect_elements:
                aspect_synonyms = get_synonyms(aspect_element)
                for sentiment_name in sentiment_synonyms:
                    for aspect_name in aspect_synonyms:
                        verbalization2L = f"{sentiment_name} {aspect_name} is {sentiment_value}."
                        verbalizations2L.append(verbalization2L)

    # Use this code to print the type 2 verbalizations to check
    # if verbalizations2L:
    #     return sorted(set(verbalizations2L))
    # else:
    #     return []

    # Return the sorted unique verbalizations
    return sorted(set(verbalizations2L)) if verbalizations2L else []

type2_verbalizations = verbalize_type2_sentiments_laptop()



# TYPE 3 SENTIMENT EXPRESSIONS

# Function to get synonyms (labels) for a class
def get_synonyms(cls):
    synonyms = []
    # Get rdfs:label
    if cls.label:
        synonyms.extend(str(label) for label in cls.label if "Mention" not in str(label))
    # Search for skos:prefLabel and skos:altLabel properties
    for prop in onto.search(iri="*prefLabel") + onto.search(iri="*altLabel"):
        values = prop[cls]
        synonyms.extend(str(v) for v in values if str(v) not in synonyms and "Mention" not in str(v))
    # Search for lex properties
    for prop in onto.search(iri="*lex"):
        values = prop[cls]
        synonyms.extend(str(v) for v in values if str(v) not in synonyms and "Mention" not in str(v))
    # Return synonyms, or fallback to class name with "Mention" removed
    if synonyms:
        return synonyms
    cleaned_name = cls.name.replace("Mention", "").lower()
    return [cleaned_name] if cleaned_name else []

# Function to get all relevant subclasses recursively, excluding those with "Property" or "Action"
def get_relevant_subclasses(cls):
    relevant = []
    for subclass in cls.subclasses():
        if "Property" in subclass.name or "Action" in subclass.name or "Sentiment" in subclass.name:
            continue
        relevant.append(subclass)
        relevant.extend(get_relevant_subclasses(subclass))
    return relevant

# Function to verbalize type 3 sentiment expressions using synonyms and instances from aspect classes and their subclasses
def verbalize_type3_sentiments():
    # Find the SentimentValue class
    sentiment_value = onto.search_one(iri="*SentimentValue")
    if not sentiment_value:
        print("Error: SentimentValue class not found.")
        print("\nAll classes in ontology for debugging:")
        for cls in onto.classes():
            print(f"- {cls.name} (IRI: {cls.iri})")
        return []

    # Type 2 aspect classes and specific subclasses
    aspect_classes = {
        'operation_mention': onto.search_one(iri="*Operation_performanceMention"),
        'price_mention': onto.search_one(iri="*PriceMention"),
        'shipping_mention': onto.search_one(iri="*ShippingMention"),
        'space_mention': onto.search_one(iri="*SpaceMention"),
        'support_mention': onto.search_one(iri="*SupportMention"),
        'quality_mention': onto.search_one(iri="*QualityMention"),
    }

    # Check for missing aspect classes
    missing_aspects = [key for key, cls in aspect_classes.items() if not cls]
    if missing_aspects:
        print(f"Error: Missing aspect classes: {', '.join(missing_aspects)}")
        print("\nAll classes in ontology for debugging:")
        for cls in onto.classes():
            print(f"- {cls.name} (IRI: {cls.iri})")
        return []

    # Defines sentiment classes (assuming they are subclasses of SentimentValue)
    sentiment_classes = {
        'fast': onto.search_one(iri="*Fast"),
        'high': onto.search_one(iri="*High"),
        'low': onto.search_one(iri="*Low"),
        'slow': onto.search_one(iri="*Slow"),
    }

    # Checks for missing sentiment classes
    missing_sentiments = [key for key, cls in sentiment_classes.items() if not cls]
    if missing_sentiments:
        print(f"Warning: Missing sentiment classes: {', '.join(missing_sentiments)}")

    # Dictionary connecting sentiment classes to aspect classes with polarity (according to definition type 3)
    sentiment_to_aspect_polarity = {
        'fast': [
            {'aspect': 'operation_mention', 'polarity': 'positive'},
            {'aspect': 'support_mention', 'polarity': 'positive'},
            {'aspect': 'space_mention', 'polarity': 'positive'},
            {'aspect': 'shipping_mention', 'polarity': 'positive'}
        ],
        'high': [
            {'aspect': 'price_mention', 'polarity': 'negative'},
            {'aspect': 'quality_mention', 'polarity': 'positive'}
        ],
        'low': [
            {'aspect': 'price_mention', 'polarity': 'positive'},
            {'aspect': 'quality_mention', 'polarity': 'negative'}
        ],
        'slow': [
            {'aspect': 'operation_mention', 'polarity': 'negative'},
            {'aspect': 'support_mention', 'polarity': 'negative'},
            {'aspect': 'space_mention', 'polarity': 'negative'},
            {'aspect': 'shipping_mention', 'polarity': 'negative'}
        ]
    }

    # Format the mappings into a list of strings using synonyms for sentiments and aspects
    formatted_list = []
    for sentiment_key, aspects in sentiment_to_aspect_polarity.items():
        # Get the sentiment class, if available
        sentiment_cls = sentiment_classes.get(sentiment_key)
        # Get synonyms for the sentiment (use class synonyms if found, else fallback to key)
        sentiment_synonyms = get_synonyms(sentiment_cls) if sentiment_cls else [sentiment_key]
        if not sentiment_synonyms:
            print(f"Warning: No synonyms found for sentiment {sentiment_key}.")
            sentiment_synonyms = [sentiment_key]  # Fallback to the key itself

        for item in aspects:
            aspect_key = item['aspect']
            polarity = item['polarity']
            # Get the ontology class for the aspect
            aspect_class = aspect_classes.get(aspect_key)
            if aspect_class:
                # Get synonyms for the aspect class
                aspect_synonyms = get_synonyms(aspect_class)
                # Generate strings for the aspect class synonyms
                for sentiment_synonym in sentiment_synonyms:
                    for synonym in aspect_synonyms:
                        formatted_list.append(f"{sentiment_synonym} {synonym} is {polarity}")
                # Get subclasses and their synonyms
                subclasses = get_relevant_subclasses(aspect_class)
                for subclass in subclasses:
                    subclass_synonyms = get_synonyms(subclass)
                    for sentiment_synonym in sentiment_synonyms:
                        for synonym in subclass_synonyms:
                            formatted_list.append(f"{sentiment_synonym} {synonym} is {polarity}")
            else:
                # Fallback to the aspect key if class not found
                for sentiment_synonym in sentiment_synonyms:
                    formatted_list.append(f"{sentiment_synonym} {aspect_key} is {polarity}")

    # # Return the formatted list, sorted and deduplicated
    # return sorted(set(formatted_list))

    # Return the sorted unique verbalizations
    return sorted(set(formatted_list)) if formatted_list else []

# Assuming verbalize_type2_sentiments_laptop and verbalize_type3_sentiments are defined elsewhere
type3_verbalizations = verbalize_type3_sentiments()

# Use this code to print the type 3 verbalizations to check
# # Call the function and print the result
# result = verbalize_type3_sentiments()
# if result:
#     print("")
#     for line in result:
#         print(line)
# else:
#     print("\nNo Type-3 Sentiment Verbalizations generated.")


# Combine the lists of all types
combined_verbalizations = type1_verbalizations + type2_verbalizations + type3_verbalizations

# Save the combined verbalizations to a JSON file
with open("verbalizations.json", "w", encoding="utf-8") as f:
    json.dump(combined_verbalizations, f, ensure_ascii=False, indent=4)

