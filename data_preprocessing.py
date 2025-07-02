# Some parts are taken from: https://github.com/QuintenvdVijver/Ontology-Augmented-Prompt-Engineering/blob/main/data%20pre-processing.py
# Run this code per step and comment out the other steps
# For each next step use the corrected dataset from the previous step
import xml.etree.ElementTree as ET
import xml.dom.minidom
import xmltodict
import json
from collections import Counter

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

# STEP 2: remove implicit or conflicting aspects in all datasets
def delete_implicit_aspects(input_path, output_path):
    #deleting the implicit aspects (target = null) from the dataset

    # Load input file
    tree = ET.parse(input_path)
    root = tree.getroot()

    #Go over all sentences in the dataset
    for review in root.findall('.//Review'):
        sentences = review.find('sentences')
        if sentences is None:
            continue
            
        # Create a list to store sentences to be removed
        sentences_to_remove = []
        
        # Looks for Opinions in each sentence
        # If no Opinions are found, sentence is put in the sentences_to_remove list
        for sentence in sentences.findall('sentence'):
            opinions_elem = sentence.find('Opinions')
            if opinions_elem is None:
                sentences_to_remove.append(sentence)
                continue
    
            opinions = opinions_elem.findall('Opinion')
 
            # Counts NULL targets and total opinions
            null_targets = sum(1 for opinion in opinions if opinion.get('target') == 'NULL')
            conflict_polarities = sum(1 for opinion in opinions if opinion.get('polarity') == 'conflict')
            total_opinions = len(opinions)
            
            # Decides what will be removed
            if null_targets == total_opinions or conflict_polarities == total_opinions:
                # All targets are NULL, mark sentence for removal
                sentences_to_remove.append(sentence)
            else:
                # Remove only Opinioins with NULL targets or conflict polarities
                for opinion in opinions[:]:  # Create a copy to modify during iteration
                    if opinion.get('target') == 'NULL'or opinion.get('polarity') == 'conflict':
                        opinions_elem.remove(opinion)
        
        # Remove marked sentences
        for sentence in sentences_to_remove:
            sentences.remove(sentence)

    # Write the modified XML to the output file
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    print(f"Dataset with implicit aspects removed saved to: {output_path}")


# Define input and output paths
# Input files are the raw 2015 and 2016 datasets and the corrected 2014 datasets after step 1
input_path = "INPUT_PATH"
output_path = "OUTPUT_PATH"

# Call the function with the previously specified paths
delete_implicit_aspects(input_path, output_path)


# STEP 3: remove intersections between training and test data from the training files
def remove_intersections(training_input_path, validation_input_path, training_output_path):

    # Exstract all sentences from the training and test data
    def extract_sentences_from_xml(input_file):
            # Parse the XML string
            root = ET.parse(input_file)
            
            # Create a set to store sentence texts
            sentences = set()
            
            # Find all sentence elements and extract their text
            for sentence in root.findall('.//sentence'):
                text_elem = sentence.find('text')
                if text_elem is not None and text_elem.text:
                    sentences.add(text_elem.text.strip())
            
            return sentences
    
    # Computes the intersections between training and test data
    def intersection(training_file, validation_file):

        # Extract sentences from both datasets
        train_sentences = extract_sentences_from_xml(training_file)
        valid_sentences = extract_sentences_from_xml(validation_file)

        # Find intersections (sentences in both training and test data)
        common_sentences = train_sentences.intersection(valid_sentences)
    
        # Return count of shared sentences
        return common_sentences
        
    # Load input files
    training_tree = ET.parse(training_input_path)
    training_root = training_tree.getroot()
    intersection = intersection(training_input_path, validation_input_path)

    # Go over all reviews and sentences
    for review in training_root.findall('.//Review'):
        sentences = review.find('sentences')
        if sentences is None:
            continue
            
        # Create a list to store sentences to remove
        sentences_to_remove = []
        
        for sentence in sentences.findall('sentence'):
            
            # If sentence text in intersection, remove sentence
            if sentence.find('text').text.strip() in intersection:
                sentences_to_remove.append(sentence)

        # Remove sentences in this list
        for sentence in sentences_to_remove:
            sentences.remove(sentence)

    # Write the modified XML to the output file
    training_tree.write(training_output_path, encoding="utf-8", xml_declaration=True)
    print(f"Training Dataset with intersections removed saved to: {training_output_path}")

# Define input and output paths
training_input_path = "TRAINING_INPUT_PATH"
validation_input_path = "TEST_INPUT_PATH"
training_output_path = "TRAINING_OUTPUT_PATH"

# Call the function with the previously specified paths
remove_intersections(training_input_path, validation_input_path, training_output_path)


# STEP 4: convert XML files to JSON files
# Read and parse the XML files
with open('INPUT_PATH', 'r') as xml_file:
    xml_data = xml_file.read()

# Convert XML to Python dictionary
dict_data = xmltodict.parse(xml_data)

# Convert dictionary to JSON
json_data = json.dumps(dict_data, indent=4)

# Save JSON to a file
with open('output.json', 'w') as json_file:
    json_file.write(json_data)

# Returns the inputted data file as a JSON file
print(json_data)






