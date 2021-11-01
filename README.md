# Email-Classifier
This repository contains a NLP Project built to classify emails as abusive or non abusive. 
The dataset contains 24656 emails with two columns : Content and Class.
Various text cleaning processes were carried out and only those models were tested which can support sparse matrices. Out of them, LinearSVC performed the best in terms of precision and recall. 
The classifier was then deployed using Streamlit.
