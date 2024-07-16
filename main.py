# https://docs.python.org/3/library/argparse.html


import argparse

from relevance_feedback import QueryExpansion

parser = argparse.ArgumentParser(description='Process command line')

# parser.add_argument('-spanbert')
# parser.add_argument('-gemini')

args = parser.parse_known_args()

[engine, engine_api_key, engine_id , gemini_api_key, r, t, q, k] = args[1]



int_k = int(k)
relevance = QueryExpansion(engine = engine[1:],
                               engine_api_key = engine_api_key,
                               engine_id = engine_id,
                               gemini_api_key = gemini_api_key,
                               r = str(r),
                               t = float(t),
                               query = q,
                               k = int_k if (int_k > 0) and (int_k <= 10) else 10,
                               )

relevance.CustomerSearch()
