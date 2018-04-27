#!/usr/bin/env python3

from __future__ import division, print_function
from wand.image import Image, Color
from PIL import Image as PI
import pytesseract
import argparse
import io
import cv2
import re
import os
import numpy as np
from skimage.transform import radon
from skimage.morphology import disk, closing
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pycorenlp import StanfordCoreNLP
try:
    # More accurate peak finding from https://gist.github.com/endolith/255291#file-parabolic-py
    from parabolic import parabolic

    def argmax(x):
        return parabolic(x, np.argmax(x))[0]
except ImportError:
    from numpy import argmax

nlp = StanfordCoreNLP('http://localhost:9000')
#from matplotlib.mlab import rms_flat
import spacy
nlp_sp = spacy.load("en")


def add_rectangle_to_image(image, rect):
    if len(image.shape)==2:
        color = 255
    else:
        color = (255, 255, 255)
    image = cv2.rectangle(image, tuple(rect[0:2]), (rect[0] + rect[2], rect[1] + rect[3]), color, thickness=-1)
    return image


def preprocess_for_image(gray, preprocess_arg):
    # check to see if we should apply thresholding to preprocess
    if preprocess_arg == "thresh":
        #gray = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        gray = cv2.medianBlur(gray, 3)
        selem = disk(1)
        gray = closing(gray, selem)
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 10)

    # make a check to see if median blurring should be done to remove
    # noise
    elif preprocess_arg == "blur":
        gray = cv2.medianBlur(gray, 3)
    return gray


def rotation_spacing(I):
    # -*- coding: utf-8 -*-
    """
    Automatically detect rotation and line spacing of an image of text using
    Radon transform
    If image is rotated by the inverse of the output, the lines will be
    horizontal (though they may be upside-down depending on the original image)
    It doesn't work with black borders
    """

    # Load file, converting to grayscale
    #I = asarray(Image.open(filename).convert('L'))
    I = I - np.mean(I)  # Demean; make the brightness extend above and below zero

    # Do the radon transform and display the result
    sinogram = radon(I)

    # Find the RMS value of each row and find "busiest" rotation,
    # where the transform is lined up perfectly with the alternating dark
    # text and white lines
    r = np.array([np.sqrt(np.mean(np.abs(line) ** 2)) for line in sinogram.transpose()])
    rotation = argmax(r)
    if rotation < 45:
        rotation = rotation + 90
    print('Rotation: {:.2f} degrees'.format(90 - rotation))
    return rotation


def ner_extraction(text):
    output = nlp.annotate(text, properties={
        'annotators': 'ner',
        'outputFormat': 'json'
    })
    return output


def cleanup(token, lower = True):
    if lower:
       token = token.lower()
    return token.strip()


def blank_word(word, conv_img, excl=0):
    rect = [int(word[6]), int(word[7]), int(word[8])-excl, int(word[9])]

    conv_img = add_rectangle_to_image(conv_img, rect)
    return(conv_img)

def templates(term):
    match = False

    if re.match(r"[^@]+@[^@]+", term) or \
        re.match("^\s*[a-zA-Z]{2}(?:\s*\d\s*){6}[a-zA-Z]?\s*$", term) or \
        re.match("^0\d{4,10}$", term) or re.match("^\d{1}[A-Z]{2}$", term):

        match = True

    return(match)


def find_names(filename, rot_fl=0):
    # load the example image and convert it to grayscale

    image_pdf = Image(filename=filename, resolution=300)

    image_jpeg = image_pdf.convert('jpeg')

    req_image = []
    conv_img_list = []
    gray_list = []
    search_terms = []
    search_terms_sp = []
    doc_text = ''
    fin_terms = ['Address', 'Administration', 'Age', 'Agree', 'Agreement', 'Allowance',
                 'Analysis', 'Annual', 'Approx', 'Assurance', 'Authority', 'Authorisation',
                 'Balanced', 'Bank', 'Benefit', 'Birth', 'Budget', 'Business',
                 'Capita', 'Capital', 'Capitalised', 'Cash', 'Centre', 'Charge', 'Choice', 'Civil',
                 'Commencement', 'Comparison', 'Conclusion', 'Condition', 'Confident', 'Confidential',
                 'Confirmation', 'Consumer', 'Contribution', 'Control', 'Critical', 'Customs',
                 'Data', 'Date', 'Death', 'Deed', 'Definition', 'Department', 'Detail', 'Direct', 'Disagree',
                 'Discretionary', 'Discuss', 'Employment', 'Emerging', 'Entitlement', 'Equity', 'European',
                 'Fact', 'FAQ', 'Feature', 'Fee', 'File', 'Final', 'Financial',
                 'Flexibility', 'Forename', 'Free', 'Full', 'Fund',
                 'General', 'Government', 'Growth', 'Guide', 'Health',
                 'Income', 'Increase', 'Identified', 'Index', 'Industry', 'Information', 'Insignificant',
                 'Insurance', 'Interest', 'International', 'Investment', 'Investor',
                 'Legal', 'Life', 'Lifetime', 'Limited', 'Lower', 'Lump',
                 'Marital', 'Member', 'Membership', 'Mobile', 'Money', 'Mutual', 'National', 'Nominated',
                 'Normal', 'Note', 'Number', 'Offer', 'Office', 'Ongoing',
                 'Option', 'Outcome', 'Partnership', 'Paying', 'Pension', 'Percentage', 'Period', 'Personal',
                 'Phone', 'Please', 'Portfolio', 'Post', 'Price', 'Profile', 'Protection', 'Purchase',
                 'Rate', 'Reason', 'Recommendation', 'Reduce', 'Reduction', 'Reference',
                 'Register', 'Registered', 'Regulation', 'Regulator', 'Report',
                 'Research', 'Request', 'Result', 'Retail', 'Retirement', 'Revenue', 'Risk',
                 'Salary', 'Saving', 'Scheme', 'Section', 'Service', 'Solution', 'Spouse', 'Stakeholder',
                 'State', 'Statement', 'Statistics', 'Status', 'Subject', 'Sum', 'Summary', 'Support', 'Surname',
                 'Tax', 'Taxation', 'Tel', 'Telephone', 'Total', 'Transfer',
                 'Trust', 'Trustee', 'Type', 'Typical', 'Typically',
                 'Unauthorised', 'Unit', 'Value', 'Version', 'Wealth', 'Yield', 'Your', 'Yours']

    for img in image_jpeg.sequence:
        img_page = Image(image=img)
        img_page.background_color = Color('white')
        img_page.alpha_channel = 'remove'
        req_image.append(img_page.make_blob('jpeg'))

    for img in req_image:
        # txt = pytesseract.image_to_string(PI.open(io.BytesIO(img)))
        conv_img = PI.open(io.BytesIO(img))
        conv_img = np.asarray(conv_img, dtype=np.uint8)

        if len(conv_img.shape) == 3:
            #conv_img = cv2.cvtColor(conv_img, cv2.COLOR_RGB2BGR)

            gray = cv2.cvtColor(conv_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = conv_img
        gray = preprocess_for_image(gray, preprocess_arg="thresh")

        # Rotate images
        if rot_fl == 1:
            rot = rotation_spacing(gray)
        else:
            rot = 90.0

        rows, cols = gray.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90 - rot, 1)
        gray = cv2.warpAffine(gray, M, (cols, rows))
        conv_img = cv2.warpAffine(conv_img, M, (cols, rows))
        page_text = pytesseract.image_to_string(gray, config='--psm 4 -c textord_heavy_nr=1')
        #print(page_text)

        doc_text = doc_text + page_text

        conv_img_list.append(conv_img)
        gray_list.append(gray)

        # NLP analysis
        nlp_result = ner_extraction(page_text)

        nlp_result_sp = nlp_sp(page_text)
        labels = set([w.label_ for w in nlp_result_sp.ents])
        in_labels = ['PERSON', 'ORG', 'GPE', 'LOC', 'FAC']

        others =[]

        for sen in nlp_result["sentences"]:
            for tok in sen['tokens']:
                #print(tok)
                if tok['ner'] == 'PERSON' or tok['ner'] == 'LOCATION' or tok['ner'] == 'ORGANIZATION' or tok[
                        'ner'] == 'MISC':
                    if tok["word"] not in search_terms and len(tok["word"]) > 1 and tok["word"] not in fin_terms \
                            and not tok["word"].islower():
                        search_terms.append(tok["word"])

                if tok['ner'] == 'O':
                    others.append(tok["word"])
                # Find emails, NINs and phone numbers
                if templates(tok["word"]):
                    search_terms.append(tok["word"])


        for label in labels:
            if label in in_labels:
                entities = [cleanup(e.string, lower=False) for e in nlp_result_sp.ents if label == e.label_]
                entities = list(set(entities))
                #print(label, entities)

                for ent in entities:
                    wds_list = re.split(' |\n', ent)
                    for wd in wds_list:
                        if wd not in search_terms and wd not in search_terms_sp and len(wd) > 1 and wd in others and not wd.islower():
                            search_terms_sp.append(wd)

    search_terms1 = []

    for term in search_terms_sp:
# and term.lower() not in doc_text
        if term not in fin_terms and term[:-1] not in fin_terms:

                    search_terms1.append(term)
        else:
            if templates(term):
                search_terms1.append(term)

    #tel = re.search("^(\+44\s?\d{4}|\(?0\d{4}\)?)\s?\d{3}\s?\d{3}$", doc_text)
    #print(tel)

    search_terms = search_terms + search_terms1
    print(search_terms)
    #Save search terms
    #text_file = open(filename[:-4] + 'terms.txt', "w")
    #text_file.write("%s" % search_terms1)
    #text_file.close()

    return search_terms, conv_img_list, gray_list


def draw_boxes(search_terms, conv_img_list, gray_list, dir, filename):
    out_file = os.path.join(dir, filename[:-4] + "_anon.pdf")
    pp = PdfPages(out_file)

    for ind in range(len(conv_img_list)):
        conv_img = conv_img_list[ind]
        gray = gray_list[ind]

        boxes = pytesseract.image_to_data(gray, config='--psm 4 -c textord_heavy_nr=1')
        # -c "textord_heavy_nr"=1 -c "textord_space_size_is_variable"=1
        #print(boxes)
        lines = boxes.split('\n')
        words = [x.split('\t') for x in lines]


        for i in range(len(lines)):
            if len(words[i]) == 12:
                if (words[i][11] in search_terms):
                    conv_img = blank_word(words[i], conv_img)

                if words[i][11] in ['Phone', 'Phone:', 'Telephone', 'Telephone:', 'Tel:', 'Tel.:']:
                    for w in range(5):
                        if any(char.isdigit() for char in words[i+w][11]):
                            conv_img = blank_word(words[i+w], conv_img)

                if (words[i][11] in ['D.o.B.', 'D.o.B', 'Birth',
                                                             'D.o.B.:', 'D.o.B:', 'Birth:', 'DoB', 'DoB:']):
                    for w in range(5):
                        if any(char.isdigit() for char in words[i+w][11]):
                            if len(words[i+w][11]) > 5:
                                char_w = int(words[i+w][8]) / len(words[i+w][11])
                                excl = int(2*char_w) + 5
                                conv_img = blank_word(words[i+w], conv_img, excl)
                            else:
                                conv_img = blank_word(words[i + w], conv_img)
                                conv_img = blank_word(words[i + w + 1], conv_img)

                if templates(words[i][11]):
                    conv_img = blank_word(words[i], conv_img)

        imgplot = plt.figure(figsize=(8.27, 11.69), dpi=300)
        ax = imgplot.add_axes([0.0, 0.0, 1.0, 1.0], frameon=False, aspect=1)
        if len(conv_img.shape) == 2:
            plt.imshow(conv_img, cmap='gray')
        else:
            plt.imshow(conv_img)
        ax.set_xticks([])
        ax.set_yticks([])
        pp.savefig()


    pp.close()


parser = argparse.ArgumentParser(description='Anonymise PDF documents.')

parser.add_argument('dir', nargs='?', help="Directory with PDFs to process")
parser.add_argument('skew', nargs='?', default=0, help="Skewness flag. Put 1 to correct skewness, 0 otherwise")


args = parser.parse_args()
if not args.dir:
        print('--Folder name is mandatory')
        sys.exit(1)

directory = os.fsencode(args.dir)

outdir = os.path.join(args.dir, 'Anonymised')
if not os.path.exists(outdir):
    os.makedirs(outdir)

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".pdf"):
        [search_terms, conv_img_list, gray_list] = find_names(os.path.join(args.dir, filename), args.skew)
        draw_boxes(search_terms, conv_img_list, gray_list, outdir, filename)
