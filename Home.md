Welcome to the NLP-Interview-preparation wiki!

# Q.1: Outline the key differences in subsampling and Negative sampling in Word2Vec Algorithm.
Ans: Let's outline the key differences between subsampling and negative sampling in the Word2Vec algorithm in steps for interview preparation:

**Subsampling in Word2Vec:**

1. **Step 1: Corpus Preparation**: Begin with a corpus of text data, which is a collection of sentences or documents.

2. **Step 2: Vocabulary Creation**: Build a vocabulary of unique words present in the corpus.

3. **Step 3: Word Frequency Calculation**: Calculate the frequency of each word in the corpus. More common words will have higher frequencies.

4. **Step 4: Subsampling Threshold**: Define a subsampling threshold for each word. This threshold is typically based on the word's frequency. The higher the frequency, the higher the threshold.

5. **Step 5: Subsampling Process**: For each word in the vocabulary, compare its frequency to the subsampling threshold. If the frequency is higher than the threshold, the word has a higher probability of being subsampled (i.e., removed from the training data). This process helps to reduce the impact of very frequent words, which might not carry as much semantic information.

6. **Step 6: Training**: Train the Word2Vec model using the subsampled training data, where some of the high-frequency words have been removed. This helps focus the model on learning from more informative words.

**Negative Sampling in Word2Vec:**

1. **Step 1: Corpus Preparation**: Begin with the same corpus of text data as in subsampling.

2. **Step 2: Vocabulary Creation**: Build a vocabulary of unique words present in the corpus.

3. **Step 3: Positive Examples**: For each word in the vocabulary, define a positive example context. This context is typically a window of words surrounding the target word.

4. **Step 4: Negative Sampling**: Here's where Negative Sampling comes into play:
   - For each positive example (target word-context pair), randomly select a small number (usually 5-20) of negative examples. These are words that don't appear in the context of the target word.
   - The idea is to train the model to distinguish between the actual context (positive) and randomly sampled noise (negative).

5. **Step 5: Model Architecture and Training**: Use Skip-gram or CBOW models and train the Word2Vec model by minimizing a loss function that distinguishes positive and negative examples.

**Key Differences:**

- **Subsampling** focuses on reducing the influence of high-frequency words by potentially removing them from the training data.
- **Negative Sampling** aims to improve the efficiency of training by focusing on a small number of positive and negative examples for each word, rather than considering all words in the vocabulary.
- Subsampling affects the data used for training, while Negative Sampling affects the training process and the loss function.
- Subsampling doesn't directly address the problem of training efficiency, while Negative Sampling explicitly addresses the computational complexity of training.

In summary, subsampling and negative sampling are two distinct techniques used in Word2Vec to improve training efficiency and the quality of word embeddings, with subsampling focusing on data reduction and negative sampling addressing computational complexity.

# Q.2: How does it determine the subsampling threshold?
Ans: The subsampling threshold in Word2Vec is determined based on the word's frequency within the training corpus. The more frequent a word is, the higher its subsampling threshold. This is done to reduce the impact of very common words on the training process and to focus more on learning representations for less frequent but potentially more informative words. Here's how the subsampling threshold is typically determined:

1. **Calculate Word Frequencies**: Count the number of times each word appears in the training corpus. These frequencies are typically measured as the raw count or the relative frequency (number of occurrences divided by the total number of words).

2. **Calculate Subsampling Probability**: The subsampling probability (P) for each word is calculated using a formula that takes its frequency into account. The formula often used is as follows:

   ```
   P(word) = 1 - sqrt(t / f(word))
   ```

   Where:
   - `f(word)` is the frequency of the word in the corpus.
   - `t` is a hyperparameter that controls the threshold. It's usually set to a small positive value, like 1e-5.

3. **Determine the Threshold**: To obtain the actual subsampling threshold, you can convert the probability into a threshold value. The threshold is usually set such that the most frequent words have lower thresholds, making them more likely to be subsampled, while less frequent words have higher thresholds, making them less likely to be subsampled.

4. **Random Selection**: During the subsampling process, a random number is generated for each word in the training data. If this random number is greater than the calculated subsampling probability for the word, the word is retained for training; otherwise, it is discarded.

By determining the subsampling threshold based on word frequencies, Word2Vec aims to balance the importance of frequent and infrequent words in the training process. Frequent words are downsampled to prevent them from dominating the training, while infrequent words are retained to capture more meaningful contextual information.

# Q.3: Explain LDA Model architecture.
Ans: Latent Dirichlet Allocation (LDA) is a generative probabilistic model used for topic modeling. It helps uncover the underlying topics within a collection of documents. Here are the steps to explain the LDA model:

1. **Collect a Corpus of Documents**:
   Gather a collection of documents that you want to analyze for underlying topics. These documents can be in the form of articles, emails, or any textual data.

2. **Preprocessing**:
   Before applying LDA, preprocess the documents by removing stopwords, punctuation, and other noise. You may also perform stemming or lemmatization to reduce words to their base forms.

3. **Create a Document-Term Matrix (DTM)**:
   Convert the collection of documents into a Document-Term Matrix. This matrix represents the frequency of terms (words) within each document. Rows correspond to documents, and columns correspond to unique terms in the corpus.

4. **Choose the Number of Topics (K)**:
   Decide on the number of topics you want to identify in the corpus. This is a crucial parameter and can significantly impact the results.

5. **Initialize LDA Model**:
   Initialize the LDA model with the DTM and the chosen number of topics (K). LDA uses a generative process to assign topics to words in documents and topics to documents.

6. **Gibbs Sampling**:
   LDA often uses Gibbs sampling or variational inference to estimate the topic distribution for each document and the word distribution for each topic. This process involves iteratively updating the topic assignments for words in documents.

7. **Iterate and Converge**:
   Continue the Gibbs sampling or variational inference process for a set number of iterations or until the model converges. Convergence indicates that the model has reached a stable state where topics are well-defined.

8. **Review Topics**:
   After convergence, you can examine the resulting topic-word distributions and document-topic distributions. This helps you understand what topics are prevalent in the corpus and how words are associated with those topics.

9. **Label and Interpret Topics**:
   Based on the words with the highest probabilities in each topic and the documents' topic proportions, assign meaningful labels to the topics. These labels should represent the themes or subjects that the topics capture.

10. **Application**:
    Use the trained LDA model to categorize new or unseen documents into topics or to explore the distribution of topics within your corpus. It can be used for tasks like document classification, recommendation systems, and content analysis.

LDA is a powerful tool for uncovering the latent topics within a collection of documents, making it valuable for various natural language processing and text analysis applications.

# Q.4: How LDA uses Gibbs sampling to estimate the topic distribution?
Ans: Latent Dirichlet Allocation (LDA) uses Gibbs sampling as a Bayesian inference technique to estimate the topic distribution for words in documents and the word distribution for topics. Here's how LDA employs Gibbs sampling to achieve this:

1. **Initialize Assignments**:
   Start with a random assignment of topics to each word in the corpus. Each word in a document is assigned to one of the K topics, where K is the predetermined number of topics.

2. **Iterate Over Documents and Words**:
   LDA performs a series of iterations through all the documents and words in the corpus. For each word in each document, it considers the current topic assignment and calculates the probability of reassigning that word to a different topic.

3. **Calculate Topic Assignment Probabilities**:
   LDA calculates two key probabilities for each word:
   
   a. **Word-to-Topic Probability**: The probability of the current word being assigned to each of the K topics, based on the topic distribution in the document and the word distribution in the topics.
   
   b. **Topic-to-Word Probability**: The probability of each topic generating the current word, based on the count of that word in the topic.

4. **Reassign Word to a Topic**:
   Using the calculated probabilities, LDA reassigns the topic for the word in question. It samples a new topic assignment for the word based on these probabilities. The choice of topic is influenced by both the current word's distribution in the topics and the topics' distribution in the document.

5. **Repeat Iterations**:
   Continue iterating through all the documents and words, repeatedly reassigning topics for words. The algorithm iterates for a set number of times or until it converges, which means the topic assignments stabilize.

6. **Collect Statistics**:
   During the Gibbs sampling process, LDA collects statistics on the number of words assigned to each topic in each document and the number of times each word is assigned to each topic. These statistics are used to update the topic distributions.

7. **Estimate Topic Distributions**:
   After the Gibbs sampling iterations, LDA uses the collected statistics to estimate the topic distribution for each document and the word distribution for each topic. These distributions are the final output of the model.

By iteratively reassigning topics to words based on probabilities, Gibbs sampling helps LDA learn the underlying topic structure of the corpus. It ensures that words in documents are more likely to be assigned to topics that are relevant to the document and that topics capture meaningful patterns in the words. This process is repeated until a stable state is reached, providing a representation of topics and their distribution across documents in the corpus.

# Q.5: Explain mathematics behind Gibbs sampling used for topic distribution in LDA.
Ans: Gibbs sampling is a Markov Chain Monte Carlo (MCMC) method used to estimate the topic distribution in Latent Dirichlet Allocation (LDA). It's a probabilistic approach, and the mathematics behind it can be quite complex, but I'll provide a simplified explanation of how Gibbs sampling works in the context of LDA.

Let's assume we have a document-term matrix (DTM) where each word in a document is assigned to one of K topics. We want to estimate the topic distribution for each word in the corpus. In Gibbs sampling, we iteratively update the topic assignments for each word while keeping the assignments for all other words fixed.

Here are the key equations and steps:

1. **Setup**:
   - The document-term matrix (DTM) contains N words in D documents.
   - There are K topics.
   - For each word w in the corpus, it is assigned a topic z, where z ranges from 1 to K.

2. **Goal**:
   - Calculate the probability of topic assignment for each word in the corpus.

3. **Calculate Word-to-Topic Probability**:
   - For each word w in the corpus, calculate the conditional probability of the word being assigned to each of the K topics, given the current topic assignments for all other words.

   ```
   P(z_i = k | z_{-i}, w_i, D) ∝ (n_{-i, k} + β) * (n_{i, w} + α) / (n_{-i, k} + V * β)
   ```

   - Where:
     - `z_i` is the topic assignment for word w_i.
     - `z_{-i}` represents the topic assignments for all other words except w_i.
     - `n_{-i, k}` is the count of words assigned to topic k, excluding word w_i.
     - `n_{i, w}` is the count of word w assigned to topic k.
     - `α` and `β` are hyperparameters that control the topic and word distributions.
     - `V` is the total vocabulary size.

4. **Sample New Topic Assignment**:
   - Using the calculated probabilities, sample a new topic assignment for word w_i. This is done stochastically based on the probabilities.

5. **Repeat for All Words**:
   - Iteratively go through all words in the corpus, updating their topic assignments.

6. **Repeat Iterations**:
   - Repeat the Gibbs sampling process for a fixed number of iterations or until the topic assignments converge.

7. **Collect Statistics**:
   - Keep track of the topic assignments for each word and the counts of words assigned to each topic.

8. **Estimate Topic Distribution**:
   - After the Gibbs sampling process, you can calculate the topic distribution for each word by averaging the topic assignments over the iterations.

This process iteratively refines the topic assignments for each word based on the topic assignments of other words and the topic-word and document-topic distributions. Over time, the model converges to a stable state where the estimated topic distributions reflect the underlying topics in the corpus.

# Q.6 : Explain LSTM Model architecture in steps.
Ans: The architecture of a Long Short-Term Memory (LSTM) model can be explained in several key steps. LSTMs are a type of recurrent neural network (RNN) that is designed to handle sequential data efficiently. Here's an overview of the LSTM model architecture in steps:

1. **Input Sequence**:
   - The LSTM model starts with an input sequence, which could be a time series, a sequence of words in a text document, or any other type of sequential data.
   - Each element of the sequence is represented as a feature vector or embedding.

2. **Input Gate**:
   - For each element in the input sequence, an input gate determines which parts of the input should be stored in the memory cell and which parts should be discarded.
   - The input gate uses a sigmoid activation function to produce values between 0 and 1 for each element in the sequence.

3. **Forget Gate**:
   - The forget gate decides what information from the previous state should be retained in the memory cell and what information should be removed.
   - It also uses a sigmoid activation function to produce forget weights for each element in the memory cell.

4. **Update Memory Cell**:
   - Using the output of the input gate and the forget gate, the LSTM updates the values in the memory cell.
   - The memory cell can be thought of as a "conveyor belt" where information can be added or removed.

5. **Output Gate**:
   - The output gate decides what information from the memory cell should be used as the output of the LSTM cell.
   - It combines the current input with the content of the memory cell and produces an output using a sigmoid activation function and a tanh activation function.

6. **Hidden State Update**:
   - The LSTM cell produces an updated hidden state. This hidden state serves as both the output of the current cell and is used as the input to the next cell in the sequence.

7. **Recurrent Connections**:
   - LSTMs have recurrent connections that allow the model to process the sequence one element at a time while maintaining information over longer sequences.

8. **Stacking LSTMs** (Optional):
   - In practice, multiple LSTM cells can be stacked on top of each other to create deeper LSTM architectures. Each cell in the stack receives the hidden state output from the previous cell.

9. **Output Layer**:
   - After processing the entire sequence, the LSTM architecture may have an output layer that produces predictions or classifications based on the information learned from the sequence.

10. **Backpropagation and Training**:
    - The LSTM is trained using backpropagation through time (BPTT) to minimize a loss function, which measures the model's prediction error. The gradients are computed and used to update the model's parameters during training.

11. **Repeat for Sequences**:
    - The steps are repeated for each element in the sequence, and the information is propagated through time. This allows the LSTM to capture and remember patterns and dependencies in the data, making it suitable for a wide range of sequential tasks.

This architecture allows LSTMs to capture long-term dependencies and handle vanishing gradient problems that are often encountered in traditional RNNs, making them powerful for tasks like natural language processing, speech recognition, time series analysis, and more.

# Q.7: How does LSTM solves vanishing gradient problem?
Ans: Long Short-Term Memory (LSTM) networks are a type of recurrent neural network (RNN) designed to address the vanishing gradient problem that can occur in traditional RNNs. Here's how LSTMs solve this problem:

1. **Introduction of the Memory Cell**: LSTMs introduce a memory cell that can store and access information over long sequences. This cell is responsible for retaining information, making it less prone to vanishing gradients.

2. **Gating Mechanisms**: LSTMs use gating mechanisms to control the flow of information into and out of the memory cell. These gates are designed to learn which information to forget, remember, or update. There are three main gates in an LSTM:

   a. **Forget Gate**: This gate decides what information to discard from the previous cell state. It uses a sigmoid activation function to produce values between 0 and 1, indicating what to keep and what to forget.

   b. **Input Gate**: This gate determines what new information should be stored in the memory cell. It combines the current input and the previous cell state, processing them through a sigmoid function and a tanh function.

   c. **Output Gate**: This gate controls what information will be passed to the output. It uses the memory cell's content, modified by a sigmoid function and a tanh function.

3. **Backpropagation Through Time (BPTT)**: During training, the error gradients are backpropagated through time. The gating mechanisms and the constant flow of information within the LSTM architecture allow for more stable gradient flow. The forget gate, in particular, plays a crucial role in preventing the vanishing gradient problem because it decides what information is carried forward and what is discarded.

By using these gating mechanisms and the memory cell, LSTMs can capture and propagate information over long sequences without the vanishing gradient problem that often plagues traditional RNNs, making them effective for various sequential tasks like natural language processing and time series prediction.