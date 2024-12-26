import easyocr
import os
import cv2
import pandas as pd
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import SpacyNlpEngine
from transformers import pipeline

# Path to the directory containing your image files
folder_path = './files'

# Path to your Excel file
excel_file = './excel.xlsx'

# Define a confidence threshold
confidence_threshold = 0.85

# Load the Excel file into a DataFrame
df = pd.read_excel(excel_file)

# Display the DataFrame
print('Rows in excel file: ', df.shape[0])

# Create an EasyOCR reader
reader = easyocr.Reader(['en'])  # Specify the languages you need, e.g., 'en' for English
# Initialize the NLP engine
nlp_engine = SpacyNlpEngine()

# Create the analyzer engine
analyzer = AnalyzerEngine(nlp_engine=nlp_engine)

# Load a pre-trained model for NER
ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

# Define name-check function for validation
def is_name(string):
    entities = ner(string)
    for entity in entities:
        if entity['entity'] == 'B-PER' or entity['entity'] == 'I-PER':
            print('True: ', string)
            return True
    
    #add here more conditions - special characters, one word, capital letters...
    print('False: ', string)
    return False

# Iterate over all files in the folder
for filename in os.listdir(folder_path):
    print(filename)
    if filename.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
        # Path to the image file
        img_path = os.path.join(folder_path, filename)
        # print(img_path)
        # Read the image using OpenCV
        img = cv2.imread(img_path)

        if img is None:
            print(f"Error: The image file {img_path} is empty or not found.")
            continue  # Skip to the next file if the image is empty

        # Resize the image (e.g., resize to 2000 pixels on the longest side)
        max_dimension = 2000
        height, width = img.shape[:2]
        if max(height, width) > max_dimension:
            scaling_factor = max_dimension / float(max(height, width))
            new_size = (int(width * scaling_factor), int(height * scaling_factor))
            img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
            cv2.imwrite(img_path, img)

        # Use EasyOCR to extract text from the image
        result = reader.readtext(img_path)
        text = ' '.join([res[1] for res in result])
                        
        # Analyze the text for PII entities
        results = analyzer.analyze(text=text, entities=["PERSON"], language="en")

        # Print the detected names
        for result in results:
            if result.entity_type == "PERSON":
                if result.entity_type == "PERSON" and result.score >= confidence_threshold:
                    detected_name = text[result.start:result.end]
                    print(f"Detected name: {detected_name} (confidence: {result.score})")

                    # Get the current number of rows 
                    new_row_index = df.shape[0] 
                    df.loc[new_row_index] = [new_row_index +1, detected_name, filename]

print('Presidio results: \n', df, '\n\n Starting NER validation')

# Filter rows where 'Name' is a valid name
valid_df = df[df['Name'].apply(is_name)]

# Reset index of the filtered DataFrame
valid_df.reset_index(drop=True, inplace=True)

for i in range(0, valid_df.shape[0]):
    valid_df.loc[i, 'Entry'] = i+1

# Print the filtered DataFrame
print('Validated excel file: \n', valid_df)

df = valid_df
df.to_excel('excel.xlsx', index=False)
print('Done.')