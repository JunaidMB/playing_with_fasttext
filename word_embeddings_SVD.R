library(tidyverse)
library(tidytext)
library(SnowballC)
library(quanteda)
library(slider)
library(widyr)
library(furrr)
library(irlba)

# Read in scandal_in_bohemia story 
scandal_in_bohemia_sentences <- read_lines(file = "~/developer/playing_with_fasttext/scandal_in_bohemia_sentences.txt")

# Represent text for modelling
# Turn it into a data frame
sib_df <- tibble(scandal_in_bohemia_sentences)

sib_df <- sib_df %>% 
  mutate(sentence_id = row_number())

# Create a document feature matrix - This is not an ideal representation
sid_dfm <- sib_df %>% 
  unnest_tokens(word, scandal_in_bohemia_sentences) %>% 
  anti_join(get_stopwords()) %>% 
  mutate(stem = SnowballC::wordStem(word)) %>% 
  count(sentence_id, stem) %>% 
  bind_tf_idf(stem, sentence_id, n) %>% 
  cast_dfm(sentence_id, stem, tf_idf)
  
# Create a tidy data structure
tidy_sib <- sib_df %>% 
  select(sentence_id, scandal_in_bohemia_sentences) %>% 
  unnest_tokens(word, scandal_in_bohemia_sentences) %>% 
  anti_join(get_stopwords()) %>% 
  group_by(word) %>% 
  filter(n() >= 10) %>% 
  ungroup()

# Create nested dataframes
nested_words <- tidy_sib %>% 
  nest(words = c(word))

# Identify windows in order to calculate the skipgram probabilities
slide_windows <- function(tbl, window_size) {
  
  skipgrams <- slider::slide(
    tbl, ~.x, .after = window_size -1, .step = 1, .complete = TRUE
  )
  
  safe_mutate <- safely(mutate)
  
  out <- map2(skipgrams, 1:length(skipgrams), ~ safe_mutate(.x, window_id = .y))
  
  out %>% 
    transpose() %>% 
    pluck("result") %>% 
    compact() %>% 
    bind_rows()
}

# Calculate Point-wise mutual information (PMI)
# PMI is the logarithm of the probability of finding 2 words together, normalised for the probability of finding each of the words alone

plan(multiprocess)

tidy_pmi <- nested_words %>% 
  mutate(words = future_map(words, slide_windows, 2)) %>% 
  unnest(words) %>% 
  unite(window_id, sentence_id, window_id) %>% 
  pairwise_pmi(word, window_id)

# Determine the word vectors using Singular Value Decomposition
tidy_word_vectors <- tidy_pmi %>% 
  widely_svd(
    item = item1,
    feature = item2,
    value = pmi,
    #nv = 21, 
    #maxit = 1000
  )

# Explore word embeddings
## Find nearest neighbours to a target word

nearest_neighbors <- function(df, token) {
  df %>%
    widely(~ . %*% (.[token, ]), 
           sort = TRUE, 
           maximum_size = NULL)(item1, dimension, value) %>%
    select(-item2)
}

tidy_word_vectors %>% 
  nearest_neighbors("woman") %>% 
  filter(value >= 0) %>% 
  head(20)

# Document Embeddings
## Treat every sentence as a document
word_matrix <- tidy_sib %>% 
  count(sentence_id, word) %>% 
  cast_sparse(sentence_id, word, n)

embedding_matrix <- tidy_word_vectors %>% 
  cast_sparse(item1, dimension, value)

doc_matrix <- word_matrix %*% embedding_matrix

dim(doc_matrix)
