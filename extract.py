import PyPDF2
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pymongo
from pymongo import MongoClient
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os
from flask import Flask, send_file, request

##########Connexion base de donnee######################
connection_string = "mongodb://localhost:27017/"
database_name = "library-next-api"
collection_name = "EmployeeRepport"

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph

def generate_report(employee_name,Email,client_name,date, comments):
    file_name = f"{employee_name}_report.pdf"
    file_path = os.path.join(REPORTS_FOLDER, file_name)  # Chemin complet du fichier dans le dossier Reports
    
    doc = SimpleDocTemplate(file_path, pagesize=letter)
    styles = getSampleStyleSheet()
    normal_style = styles['Normal']
    comment_style = ParagraphStyle(name='CommentStyle', parent=normal_style, fontSize=12)

    content = []

    content.append(Paragraph(f"<b>Rapport sur l'employé :</b> {employee_name}", normal_style))
    content.append(Paragraph(f"<b>Email de l'employé :</b> {Email}", normal_style))
    content.append(Paragraph(f"<b>Date :</b> {date}", normal_style))
    content.append(Paragraph(f"<b>De la part de :</b> {client_name}", normal_style))
    content.append(Paragraph("<b>Commentaires :</b>", normal_style))

    for comment in comments.split("\n"):
        content.append(Paragraph(comment, comment_style))

    doc.build(content)
    return file_name
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_FOLDER = os.path.join(BASE_DIR, 'Reports')  
def download_file(filename):
    # Chemin complet vers le fichier à télécharger
    file_path = os.path.join(REPORTS_FOLDER, filename)
    # Vérifie si le fichier existe
    if os.path.exists(file_path):
        # Renvoie le fichier pour téléchargement
        return send_file(file_path, as_attachment=True)
    else:
        return "Fichier non trouvé", 404


def extract_pdf_content(pdf_path):
    with open(pdf_path, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        text_content = ''
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            text_content += page_text
    return text_content

# def get_reports_by_employee(employee_name):
#     return collection.find({"employee_name": employee_name})

# Example usage
# pdf_path = "khouloud-ghourabii_report.pdf"  # Replace with the actual path to your PDF file
# extracted_text = extract_pdf_content(pdf_path)
# print(extracted_text )

##############Analyse sentiment ##################################

#Chargement du tokenizer et du modèle Sentibert pré-entraîné
tokenizer = AutoTokenizer.from_pretrained("tblard/tf-allocine")

try:
    tokenizer = AutoTokenizer.from_pretrained("tblard/tf-allocine", use_fast=True)
except Exception as e:
    print("Failed to load tokenizer:", str(e))

def predict_sentiment(text):
    max_length = 514
    if type(text) == float:
        # Handle float values appropriately
        return [0.5, 0.5]  # Replace with desired behavior for float inputs
    elif len(text) > max_length:
        text = text[:max_length]

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).tolist()[0]

    # Ensure that the probabilities sum to 1
    assert abs(sum(probabilities) - 1.0) < 0.0001

    return probabilities

