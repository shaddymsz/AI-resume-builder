from transformers import DistilBertTokenizer, pipeline
from sklearn.metrics.pairwise import cosine_similarity
from pdfminer.high_level import extract_text as pdf_extract_text
from docx import Document
from flair.data import Sentence
from flair.models import SequenceTagger
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
import numpy as np
import re

# Load models
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
ner_model = pipeline('ner', model='dbmdz/bert-large-cased-finetuned-conll03-english', aggregation_strategy="simple")
flair_ner_model = SequenceTagger.load('flair/ner-english')
sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
keybert_model = KeyBERT()

# Load T5 model for text rephrasing
from transformers import T5Tokenizer, T5ForConditionalGeneration
t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-base")

def clean_keyword(keyword):
    """
    Clean and correct keywords by removing unwanted characters and common errors.
    """
    # Remove unwanted characters and fix common typos
    keyword = re.sub(r'[^a-zA-Z0-9]', '', keyword)  # Remove non-alphanumeric characters
    keyword = keyword.lower()  # Normalize to lowercase
    return keyword

def filter_tokenized_words(keywords):
    """
    Filters out subword tokens and merges them back correctly.
    Removes words with '##' and combines them with surrounding tokens.
    """
    clean_keywords = []
    for keyword in keywords:
        keyword = clean_keyword(keyword)
        if keyword.startswith("##"):
            if clean_keywords:
                clean_keywords[-1] += keyword[2:]
        else:
            clean_keywords.append(keyword)
    return list(set(clean_keywords))  # Return unique keywords

def extract_keywords(text):
    """
    Use NER and KeyBERT to extract important entities and keywords from the text.
    Post-process the output to remove subword tokens and clean keywords.
    """
    # NER with Hugging Face's transformers
    ner_results = ner_model(text)
    keywords = {entity['word'] for entity in ner_results}
    
    # NER with Flair
    flair_sentence = Sentence(text)
    flair_ner_model.predict(flair_sentence)
    keywords.update([entity.text for entity in flair_sentence.get_spans('ner')])

    # Keyword Extraction with KeyBERT
    keybert_keywords = keybert_model.extract_keywords(text)
    for keyword, _ in keybert_keywords:
        keywords.add(keyword)

    # Filter out subword tokens and merge them, then clean the keywords
    return filter_tokenized_words(list(keywords))

def get_text_similarity(text1, text2):
    """
    Calculate similarity between two texts using cosine similarity on SentenceTransformer embeddings.
    """
    embeddings1 = sentence_model.encode([text1])
    embeddings2 = sentence_model.encode([text2])

    similarity_score = cosine_similarity([embeddings1], [embeddings2])[0][0]
    return similarity_score

def generate_resume_update(resume_text, job_description):
    """
    Use T5 to generate a better version of the resume based on the job description.
    """
    input_text = f"match resume to job: {job_description} </s> resume: {resume_text}"
    
    # Tokenize input
    input_ids = t5_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    # Generate output from T5
    output_ids = t5_model.generate(input_ids, max_length=512, num_beams=4, early_stopping=True)
    
    # Decode the output
    generated_text = t5_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return generated_text

def update_resume_content(resume_text, job_description):
    """
    Update resume content based on job description using extracted keywords and similarity scores.
    Highlight changes by putting *** around them.
    """
    resume_keywords = extract_keywords(resume_text)
    job_keywords = extract_keywords(job_description)

    # Use T5 to tailor resume to job description
    updated_resume = generate_resume_update(resume_text, job_description)

    # Add missing job keywords to the resume
    for keyword in job_keywords:
        if keyword not in resume_keywords:
            updated_resume += f'\n***Relevant Skill: {keyword}***'

    return updated_resume

def extract_text_from_file(filepath):
    """
    Extract text from a file (PDF or DOCX).
    """
    if filepath.endswith('.pdf'):
        return pdf_extract_text(filepath)
    elif filepath.endswith('.docx'):
        doc = Document(filepath)
        return '\n'.join([para.text for para in doc.paragraphs])
    return ''
