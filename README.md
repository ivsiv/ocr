# Image Text Extraction and PII Detection

This project uses EasyOCR and NLP models to extract text from images and detect personally identifiable information (PII) such as names. The detected names are validated using a Named Entity Recognition (NER) model and stored in an Excel file.

## Requirements

- Python 3.12
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- OpenCV
- pandas
- [Presidio Analyzer](https://github.com/microsoft/presidio)
- transformers

## Setup

1. Install the required packages:

   ```bash
   pip install easyocr opencv-python-headless pandas presidio-analyzer transformers
   ```

2. Prepare your images and place them in the ./files directory.

3. Create an Excel file named excel.xlsx in the project root directory with the following structure:

Usage

Run the script to process the images, extract text, and detect PII:

```bash
python script.py
```

Features

    Extracts text from images using EasyOCR.

    Identifies and validates names using Presidio Analyzer and NER model.

    Saves the validated names and corresponding filenames in an Excel file.

This project is licensed under the MIT License.
