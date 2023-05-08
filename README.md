# cs5293sp23-project3
Name: Anish Sunchu     OU ID: 113583802

### Project Description
The goal of Project 3 is to use text analysis techniques to investigate themes and similarities for smart cities with the use of cluster analysis, topic modeling, and summarization. The project involves downloading and cleaning PDF documents related to the 2015 Smart City Challenge. The project has two phases:

Phase 1 involves data collection and preprocessing, where the PDF documents are downloaded and converted to text format, and then cleaned to remove stop words, punctuations, and other unwanted characters.

Phase 2 involves exploring clustering models to group similar smart cities together based on their characteristics, performing topic modeling to derive meaning from the clusters, and extracting a summary and keywords for each smart city document to gain insights into their main features and benefits.

The project aims to help stakeholders understand smart cities using data from the 2015 Smart City Challenge and to provide them with a way to compare their own smart city to others based on themes and concepts.

### **Funtions:**
**preprocess_text():**
* The preprocess_text() function performs several text preprocessing steps on the input text to make it more suitable for natural language processing tasks:
* contractions.fix(input_text): This step expands any contractions in the input text, such as "can't" to "cannot".
* re.sub(r'[^a-zA-Z0-9\s]', '', input_text): This step removes any special characters and symbols that are not alphanumeric or whitespace characters.
* word_tokenize(input_text): This step tokenizes the input text into a list of words.
* set(stopwords.words('english')): This step loads a set of stop words for the English language from the stopwords corpus in NLTK. These are common words such as "the", "and", and "a", which are often removed from text during preprocessing.
* [token for token in tkns if token.lower() not in stp_wrds]: This step removes any stop words from the tokenized list of words.
* WordNetLemmatizer(): This step initializes a lemmatizer from NLTK's WordNet corpus, which will reduce words to their base form (e.g. "going" becomes "go").
* [lmtzr.lemmatize(token) for token in tkns]: This step applies the lemmatizer to each word in the tokenized list.
* ' '.join(tkns): This step joins the list of processed tokens into a string with whitespace between each word.

### **Test Functions:**
**test_preprocess_text():**
This is a test method for the preprocess_text() function. It tests if the function correctly preprocesses the input text and produces the expected output. Here's a breakdown of the method:
input_text = "I'm gonna go to the store and buy some milk. Wanna come?" : This is the input text to be preprocessed.
expected_output = "gon na go store buy milk wan na come" : This is the expected output after the input text has been preprocessed.
self.assertEqual(project3.preprocess_text(input_text), expected_output) : This line calls the preprocess_text() function with the input text and checks if the output matches the expected output. If the two outputs are the same, the test passes. If they are different, the test fails.

### **Bugs:**

The PdfReader class from the PyPDF2 library is not imported correctly. It should be from PyPDF2 import PdfFileReader.
The preprocess_text function should convert the input text to lowercase before checking if each token is in the stop words set. The set function is case sensitive by default.
The data variable in the loop that calculates clustering scores is overwritten with a reshaped version of itself, causing a ValueError when fit_predict is called on it later in the loop.
The DBSCAN clustering method is used even if the number of unique labels is less than 2, which results in an error. The condition should be if the number of unique labels is greater than 1.

### **Assumptions:**

The PDF document contains text data that can be processed and analyzed for clustering. If the PDF contains only images or scanned pages, the script will not work.
The script assumes that the PDF document contains data related to a single city, based on the filename. If the filename does not include the name of a city, the results may not be meaningful.
The optimal number of clusters is determined based on the maximum silhouette score for K-means and hierarchical clustering. There is no guarantee that this is the best choice for clustering the data, and the user may need to experiment with different values of k to find the most appropriate clustering.