A README file including the following information:

A list of all the files that you are submitting
proj2.tar.gz, which contains:
main.py
relevance_feedback.py
spacy_help_functions.py
spanbert.py
README.pdf
Transcript.6111.Gates.spanbert.txt
Transcript.6111.Gates.gemini.txt

A clear description
1. go into the directory
2. active your environment: source dbproj/bin/activate
3. run:
python3 main.py -spanbert Google_Custom_Search_Engine_JSON_API_Key Engine_ID 2 0.7 "bill gates microsoft" 10
python3 main.py -gemini Google_Custom_Search_Engine_JSON_API_Key Engine_ID 2 0.7 "bill gates microsoft" 10

A clear description of the internal design
1. CustomerSearch(): top level function. It performs initial search iterate the pages and the sentences in each
page. For each sentence, it calls process_spanbert_sentence() or process_gemini_sentence(),
depending on the request.
2. process_spanbert_sentence() generates candidate pairs and filter the candidate pairs to match which
pairs match the relation we want. We store the confidence for each pair in dict X. When we meet a same pair
again, we only store one with a higher confidence. We use the confidence to order the pairs and decide on
new query if needed.
3. process_gemini_sentence() uses prompt_examples to request from Gemini what pairs match the
relationship requested. Our gemini logic uses helper function get_gemini_completion() to invoke the
Gemini service.

A detailed description of how to carry out
1. To retrieve each webpage we used the reference code at https://github.com/googleapis/google-apipython-
client/blob/main/samples/customsearch/main.py That is, we invokes req = requests.get(url,
headers) On failure to load a page, we just continue on to the next page
2. To extract the actual plain text, we invoke our helper function text_from_html(), which calls
BeautifulSoup() on the body. The helper function then extracts and joins everything in the "soup" where
text=True and the text is visible.
3. We truncate the text using the python idiom [:10000]
4. We call spaCy to split the text inot sentences and named entities by calling doc = self.nlp(plain_text)
5. In process_spanbert_sentence() we create candidate pairs by matching the tokens in each sentence. In
process_gemini_sentence() we use specially constructed (and tested) prompts to extract matching pairs
from each sentence (if any).
6. process_spanbert_sentence() identifies the tuples that have an associated extraction confidence of at
least t and adds them to set X. process_gemini_sentence() adds all the tuples that have been extracted to
set X. It uses a hard-coded confidence value of 1.0 for all tuples.
