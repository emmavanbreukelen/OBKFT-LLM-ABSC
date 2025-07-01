# Some parts are taken from: https://github.com/QuintenvdVijver/Ontology-Augmented-Prompt-Engineering/blob/main/data%20pre-processing.py
import xml.etree.ElementTree as ET
import xml.dom.minidom

# STEP 1: convert the 2014 dataset to the correct format (the same as 2015 and 2016)
def convert_SE2014_to_correct_format(semeval14_path, output_path):
    tree = ET.parse(semeval14_path)
    root = tree.getroot()

    # Creates new XML hierarchy
    new_root = ET.Element("Reviews")
    review = ET.SubElement(new_root, "Review")
    sentences = ET.SubElement(review, "sentences")
    
    # Go over all sentences in the dataset
    for sentence in root.findall("sentence"):
        sentence_id = sentence.get("id")
        text = sentence.find("text").text

        # Create new sentence subroot element
        new_sentence = ET.SubElement(sentences, "sentence", id=sentence_id)
        ET.SubElement(new_sentence, "text").text = text
        
        # Create new opinion subroot element
        opinions = ET.SubElement(new_sentence, "Opinions")

        # Convert <aspectTerms> into <Opinions>
        aspect_terms = sentence.find("aspectTerms")
        if aspect_terms is not None:
            # Case 1: aspectTerm is present
            for aspect in aspect_terms.findall("aspectTerm"):
                target = aspect.get("term")
                polarity = aspect.get("polarity")
                from_idx = aspect.get("from")
                to_idx = aspect.get("to")
                # Use 'from' instead of 'from_' and exclude category
                ET.SubElement(opinions, "Opinion", target=target, polarity=polarity, **{"from": from_idx, "to": to_idx})
        else:
            # Case 2: no aspectTerm is present, set target = NULL
            ET.SubElement(opinions, "Opinion", target="NULL")

    # Convert the ElementTree to a nicely printed string
    rough_string = ET.tostring(new_root, 'utf-8')
    reparsed = xml.dom.minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="    ", encoding="utf-8").decode("utf-8")
    
    # Write the XML to the output file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(pretty_xml)
    print(f"Converted SemEval14 dataset saved to: {output_path}")

# Define input and output paths
semeval14_path = "SEMEVAL_2014_PATH"
output_path = "CORRECTED_2014_PATH"

# Call the function with the previously specified paths
convert_SE2014_to_correct_format(semeval14_path, output_path)

# STEP 2: remove implicit aspects in all datasets








# STEP 3: remove intersections between training and test data from the training files



# STEP 4: convert XML files to JSON files







