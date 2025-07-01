# Ontology-Based Knowledge Fine-Tuning Large Language Models for Aspect-Based Sentiment Classification
This code can be used to knowledge fine-tune LLMs for ABSC using the knowledge from verbalized domain ontologies.

## Before running the code
- Set up the programming environment:
  - Set up your Google Colab (or something similar) with Python 3.10
  - Install the required packages: `pip install -r requirements.txt`
- Set up the data:
  - The unprocessed SemEval datasets can be found in `Data/Raw SemEval Data`
  - Using the `data_preprocessing.py` file, the data is pre-processed as follows:
    1. For the 2014 data, the XML structure is converted to the XML structure of the 2015/2016 datasets
    2. Implicit aspects are removed from the data
    3. Intersections between training and test data are removed from the training data
    4. The data is converted from an XML file to a JSON file


## Ontology verbalization
1. Open the `verbalization.py` file
2. The raw domain ontologies can be found in `Data/Raw Domain Ontologies`
3. Fill in the file path of the domain ontology you want to verbalize

This will create a JSON file where all three sentiment expression types are verbalized.


## Fine-Tuning the LLMs
1. Open the `finetuning.py` file
2. Fill in your HuggingFace password and Wandb password
3. Fill in the respective file paths of the verbalized domain ontology, test dataset, and (for few-shot evaluation) the training dataset
4. Fill in the HuggingFace file path of the LLM you want to evaluate

After following these instructions and running the code, you should be able to the three evaluation metrics (accuracy, weighted F1, and macro F1) for all four evaluation approaches (zero-shot with and without fine-tuning, three-shot with and without fine-tuning).

## References for the data
- Schouten, K., Frasincar, F., and de Jong, F. (2017). Ontology-enhanced aspect-based sentiment analysis. In 17th International Conference on Web Engineering (ICWE 2017), LNCS, pages 302-320. Springer.
- Zhuang, L., Schouten, K., and Frasincar, F. (2020). SOBA: Semi-automated ontology builder for aspect-based sentiment analysis. Journal of Web Semantics 60, 100544.
- Pontiki, M., Galanis, D., Papageorgiou, H., Androutsopoulos, I., Manandhar, S., Al-Smadi, M., Al-Ayyoub, M., Zhao, Y., Qin, B., De Clercq, O., et al. (2016). Semeval-2016 task 5: Aspect based sentiment analysis. In 10th International Workshop on Semantic Evaluation (SemEval2016), pages 19-30. ACL.
- Pontiki, M., Galanis, D., Papageorgiou, H., Manandhar, S., and Androutsopoulos, I. (2015). Semeval-2015 task 12: Aspect based sentiment analysis. In 9th International Workshop on Semantic Evaluation (SemEval2015), pages 486-495. ACL.
- Pontiki, M., Galanis, D., Pavlopoulos, J., Papageorgiou, H., Androutsopoulos, I. and Manandhar, S., (2014). Semeval-2014 task 4: Aspect based sentiment analysis. In 8th International Workshop on Semantic Evaluation (SemEval2014), pages 27-35. ACL.

## References for the SemEval data pre-processing code
- https://github.com/QuintenvdVijver/Ontology-Augmented-Prompt-Engineering

