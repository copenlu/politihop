# PolitiHop

This repo contains the **PolitiHop** dataset introduced in the paper ["Multi-Hop Fact Checking of Political Claims"](https://arxiv.org/pdf/2009.06401.pdf) and the accompanying experiments.

In this project, we study how to perform more complex claim verification on naturally occurring claims with multiple hops over evidence chunks. We first construct a small annotated dataset, **PolitiHop**, of reasoning chains for claim verification. We then compare the dataset to other existing multi-hop datasets and study how to transfer knowledge from more extensive in- and out-of-domain resources to PolitiHop. We find that the task is complex, and achieve the best performance using an architecture that specifically models reasoning over evidence chains in combination with in-domain transfer learning.

## Dataset format

Each rown in the dataset splits represents one instance and contains the following tab-separated columns:

- article_id - article id corresponding to the id of the claim in the LIAR dataset
- statement	- the text of the claim
- author - the author of the claim
- ruling - a comma-separated list of the sentences in the long ruling report (this excludes sentences from the summary in the end)
- url_sentences - a comma-separated list of the URLs ids as returned from the PolitiFact API
- relevant_text_url_sentences	- the urls that are actually relevant to the selected evidence sentences
- politifact_label - label assigned to the claim by PolitiFact fact-checkiers
- annotated_evidence - a json list of the evidence chains (keys) and the sentences that belong to the chain (value, which is a list of sentence ids from the ruling)
- annotated_label - label annotated by annotators of PolitiFact - True, False, Half-True

## IN PROGRESS: The code for the experiments is currently being prepared!
