import spacy
from collections import defaultdict

import numpy as np
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
from bs4 import BeautifulSoup
from bs4.element import Comment
from spanbert import SpanBERT
from spacy_help_functions import get_entities, create_entity_pairs
import google.generativeai as genai
from urllib.error import HTTPError
import time

# https://www.cs.columbia.edu/~gravano/cs6111/Proj2/example_relations.py
entities_of_interest = ["ORGANIZATION", "PERSON", "LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"]


spacy2bert = {
        "ORG": "ORGANIZATION",
        "PERSON": "PERSON",
        "GPE": "LOCATION",
        "LOC": "LOCATION",
        "DATE": "DATE"
        }

bert2spacy = {
        "ORGANIZATION": "ORG",
        "PERSON": "PERSON",
        "LOCATION": "LOC",
        "CITY": "GPE",
        "COUNTRY": "GPE",
        "STATE_OR_PROVINCE": "GPE",
        "DATE": "DATE"
        }

r_lookup = {"1": "Schools_Attended",
            "2": "Work_For",
            "3": "Live_In",
            "4": "Top_Member_Employees"}

relation_dict = {
    "1": "per:schools_attended",
    "2": "per:employee_of",
    "3": "per:cities_of_residence",
    "4": "org:top_members/employees",
}

relation_entity_dict = {
    "1": (["PERSON"], ["ORGANIZATION"]),
    "2": (["PERSON"], ["ORGANIZATION"]),
    "3": (["PERSON"], ["LOCATION", "CITY", "COUNTRY", "STATE_OR_PROVINCE"]),
    "4": (["ORGANIZATION", ["PERSON"]]),
}

relationship_examples = {"1": '["Jeff Bezos", "Schools_Attended", "Princeton University"]',
                        "2": '["Alec Radford", "Work_For", "OpenAI"]',
                        "3": '["Mariah Carey", "Live_In", "New York City"]',
                        "4": '["Nvidia", "Top_Member_Employees", "Jensen Huang"]'}

prompt_examples = {"1": 'Given the sentence below, if you can find a person and a school attended, give me another example like ["Jeff Bezos", "Schools_Attended", "Princeton University"].  If not, print "No such relation."',
                    "2": 'Given the sentence below, if you can find a person and an organization the person founded or works for, give me another example like ["Alec Radford", "Work_For", "OpenAI"].  If not, print "No such relation."',
                    "3": 'Given the sentence below, if you can find a person and a location the person lived in, give me another example like ["Mariah Carey", "Live_In", "New York City"].  If not, print "No such relation."',
                    "4": 'Given the sentence below, if you can find a person who is a founder or co-founder or establisher or CEO or chairperson or top employee of an organization, print the person and the organization in the format ["organization", "Top_Member_Employees", "person name"].  If no people and no organizations in the sentence print "No such relation."',
                  }

headers = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Max-Age': '3600',
    'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0'
}

# Gemini parameters
# Documentation: https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/gemini
model_name = 'gemini-pro'
max_tokens = 100
temperature = 0.2
top_p = 1
top_k = 32

# Scrape visible webpage text
# Reference https://stackoverflow.com/questions/1936466/how-to-scrape-only-visible-webpage-text-with-beautifulsoup
def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

def text_from_html(body):
    soup = BeautifulSoup(body, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)
    return u" ".join(t.strip() for t in visible_texts)

# reference: https://www.cs.columbia.edu/~gravano/cs6111/Proj2/gemini_helper_6111.py
# Generate response to prompt
def get_gemini_completion(prompt, model_name, max_tokens, temperature, top_p, top_k):

    # Initialize a generative model
    model = genai.GenerativeModel(model_name)

    # Configure the model with your desired parameters
    generation_config=genai.types.GenerationConfig(
        #max_output_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k
    )

    # Generate a response
    response = model.generate_content(prompt, generation_config=generation_config)

    return response.text

class QueryExpansion(object):

    def __init__(self, engine, engine_api_key, engine_id , gemini_api_key, r, t, query, k):
        self.engine = engine
        self.engine_api_key = engine_api_key
        self.engine_id = engine_id
        self.gemini_api_key = gemini_api_key
        self.r = r
        self.t = t
        self.query = query.strip().split()
        self.q = query
        self.k = k
        self.result = []
        self.X = set()
        self.nlp = spacy.load("en_core_web_lg")
        if engine == "spanbert":
            self.spanbert = SpanBERT("./pretrained_spanbert")
        else:
            genai.configure(api_key=gemini_api_key)

    def process_spanbert_sentence(self, isentence, sentence, X, extracted_annotations_count, relations_extracted_count):
        sentence_entity_pairs = create_entity_pairs(sentence, entities_of_interest)
        candidate_pairs = []
        for ep in sentence_entity_pairs:
            #actual_relation = relation_dict.get(relation)
            #if actual_relation:
            subject_types, object_types = relation_entity_dict[self.r]
            if (ep[1][1] in subject_types) and (ep[2][1] in object_types):
                candidate_pairs.append({"tokens": ep[0], "subj": ep[1], "obj": ep[2], "sent": isentence})  # Subject and Object text
        if (candidate_pairs == []):
            return X, extracted_annotations_count, relations_extracted_count
        relation_preds = self.spanbert.predict(candidate_pairs)
        for relation in zip(relation_preds, candidate_pairs):
            relation_pred, candidate_pair = relation
            subj = candidate_pair['subj'][0]
            obj = candidate_pair['obj'][0]
            confidence = relation_pred[1]
            if relation_pred[0] == relation_dict[self.r]:
                extracted_annotations_count += 1
                print( "                === Extracted Relation ===")
                print(f"                Input tokens: {candidate_pair['tokens']}")
                print(f"                Output Confidence: {relation_pred[1]} ; Subject: {candidate_pair['subj'][0]} ; Object: {candidate_pair['obj'][0]} ;")
                if confidence < self.t:
                    print("                Confidence is lower than threshold confidence. Ignoring this.")
                else:
                    if (subj, obj) not in X or confidence > X[(subj, obj)]:
                        relations_extracted_count += 1
                        X[(subj, obj)] = confidence
                        print("                Adding to set of extracted relations")
                    else:
                        print("                Duplicate with lower or equal confidence than existing record. Ignoring this.")
                print("                ==========")
        return X, extracted_annotations_count, relations_extracted_count


    def process_gemini_sentence(self, isentence, sentence, X, extracted_annotations_count, relations_extracted_count):
        relationship_example = relationship_examples[self.r]
        prompt_text = f"""{prompt_examples[self.r]}
            sentence: {sentence}
            """
        try:
            response_text = get_gemini_completion(prompt_text, model_name, max_tokens, temperature, top_p, top_k)
        except Exception as e:
            return X, extracted_annotations_count, relations_extracted_count
        if response_text != "No such relation.":
            subject, relationship, object = response_text[2:-2].split('", "')
            extracted_annotations_count += 1
            print( "                === Extracted Relation ===")
            print( "                Sentence:  ", sentence)
            print(f"                Subject: {subject} ; Object: {object} ;")
            if (subject, object) in X:
                print("                Duplicate. Ignoring this.")
            else:
                relations_extracted_count += 1
                X[(subject, object)] = 1.0
                print("                Adding to set of extracted relations")
            print( "                ==========")



        return X, extracted_annotations_count, relations_extracted_count


    def CustomerSearch(self):
        print('____')
        print('Parameters:')
        print(f'Client key      = {self.engine_api_key}')
        print(f'Engine key      = {self.engine_id}')
        print(f'Gemini key      = {self.gemini_api_key}')
        print(f'Method          = {self.engine}')
        print(f'Relation        = {r_lookup[self.r]}')
        print(f'Threshold       = {self.t}')
        print(f'Query           = {self.q}')
        print(f'# of Tuples     = {self.k}')
        print('Loading necessary libraries; This should take a minute or so ...)')

        query = self.q
        iteration_num = 0
        processed_urls = set()
        processed_queries = {self.q}
        X = {}  # a dict is equivalent to a set with an attached value per entry
        while (len(X) <= self.k):
            print(f"=========== Iteration: {iteration_num} - Query: {query} ===========")
            res = None
            seconds = 0
            # https://github.com/googleapis/google-api-python-client/blob/main/samples/customsearch/main.py
            service = build("customsearch", "v1", developerKey=self.engine_api_key)
            while res == None or seconds > 0:
                try:
                    res = (service.cse().list(q=query, cx=self.engine_id,).execute())
                    seconds = 0
                except HTTPError as http_error:
                    if http_error.status == 429:
                        seconds = int(res.headers["Retry-After"])
                        print(f"HTTP429: Exceeded queries per minute per user.  Need to wait {seconds} seconds.")
                        time.sleep(seconds)
                    else:
                        raise http_error

            items = res.get('items', [])

            print("len(items)",len(items))
            # display its title, URL, description and fileformat returned by Google.
            for i, item in enumerate(items):
                # https: // developers.google.com / custom - search / v1 / reference / rest / v1 / Search
                url = item.get('link', '')
                if url not in processed_urls:
                    processed_urls.add(url)
                    extracted_annotations_count = 0
                    relations_extracted_count = 0
                    print(f"URL ( {i+1} / 10): {url}")

                    print("        Fetching text from url ...")
                    try:
                        req = requests.get(url, headers)
                    except Exception as ex:
                        print (f"URL {url} failed on Exception [{ex}].  Continuing on to next url.")
                        continue
                    plain_text = text_from_html(req.content)[:10000]
                    webpage_length = len(plain_text)
                    print(f"        Webpage length (num characters): {webpage_length}")
                    print("        Annotating the webpage using spacy...")
                    doc = self.nlp(plain_text)
                    print(f"        Extracted {len(list(doc.sents))} sentences. Processing each sentence one by one to check for presence of right pair of named entity types; if so, will run the second pipeline ...")
                    for isentence, sentence in enumerate(doc.sents): # isentence is index
                        if isentence % 5 == 0:
                            print("\n        Processed {} / {} sentences".format(isentence, len(list(doc.sents))))

                        if self.engine == "spanbert":
                            X, extracted_annotations_count, relations_extracted_count = self.process_spanbert_sentence(isentence, sentence, X, extracted_annotations_count, relations_extracted_count)
                        else:
                            X, extracted_annotations_count, relations_extracted_count = self.process_gemini_sentence(isentence, sentence, X, extracted_annotations_count, relations_extracted_count)


                    print(f"        Extracted annotations for  {extracted_annotations_count}  out of total  {len(list(doc.sents))}  sentences")
                    print(f"        Relations extracted from this website: {relations_extracted_count} (Overall: {extracted_annotations_count} )")
            iteration_num += 1
            sorted_X = sorted(X.items(), key=lambda x: -x[1])
            fresh_new_queries = [x[0][0] + " " + x[0][1] for x in sorted_X if x[0][0] + " " + x[0][1] not in processed_queries]
            if len(fresh_new_queries) == 0:
                break
            query = fresh_new_queries[0]
            processed_queries.add(query)

        print(f"================== ALL RELATIONS for {relation_dict[self.r]} ( {len(X)} ) =================")
        sorted_X = sorted(X.items(), key=lambda x: -x[1])
        for item in sorted_X:
            confidence = item[1]
            person = item[0][0]
            location = item[0][1]
            confidence_str = f"Confidence: {confidence}\t\t|  " if self.engine == "spanbert" else ""
            print(f"{confidence_str}Subject: {person}\t\t| Object: {location}")
        print(f"Total # of iterations = {iteration_num}")
        exit(1)
        # return self
