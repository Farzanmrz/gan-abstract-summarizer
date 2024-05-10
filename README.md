# GANs and Transformers in Abstractive Text Summarization

## Introduction

This notebook presents the construction and demonstration of an Abstract Text Summarizer that generates summaries from CNN/Daily Mail news pieces using a GAN architecture

### Data or Knowledge Source

The CNN/DailyMail dataset, which we will utilize in this study, consists of human-generated abstractive summary bullets derived from news stories on the CNN and Daily Mail websites. These summaries are structured as questions (with one entity obscured) corresponding to stories that serve as passages from which the system must infer and fill in the blanks. The dataset includes tools for crawling, extracting, and generating passage and question pairs from these websites.
The dataset has the following article

- “By . Associated Press . PUBLISHED: . 14:11 EST, 25 October 2013 . | . UPDATED: . 15:36 EST, 25 Octob…”

This article is mapped to the summary

- “Bishop John Folda, of North Dakota, is taking time off after being diagnosed . He contracted the inf…”

The example article above is what will be passed to our generator for it to generate a summary, which will be compared with the summary in the dataset above in the discriminator.

The provided scripts structure the dataset, which comprises 286,817 training pairs, 13,368 validation pairs, and 11,487 test pairs. On average, the source documents in the training set contain 766 words spread across 29.74 sentences, while the summaries are typically 53 words and 3.72 sentences long. The complete dataset is segmented into three primary divisions: training, validation, and testing, with an overall size of 1.37 GB. 

During the project development phase, contingent upon time availability, we aim to initially surpass the benchmarks established in the reviewed paper using the same dataset. Subsequently, we plan to enhance the dataset by incorporating a diverse array of text sources and their corresponding summaries to ensure the model’s generalizability. For instance, we intend to include datasets comprising book plots, Reddit posts, and Amazon product descriptions, along with their summaries. This expansion will enrich the training data and enable the model to handle a wider variety of text types, thereby improving its applicability and robustness across different domains.

