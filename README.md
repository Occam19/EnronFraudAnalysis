By: Gurpal Sandhu

Identifying Fraud and POIs from Email & Financial Data

The Enron scandal was a high-profile discovery of illegal accounting practices at Enron Corp., a very large Texas based energy company started by Kenneth Lay and Jeffrey Skilling in 1985. That years, the SEC began to investigate their complex accounting that was confusing both shareholders and analysts alike. Jeffery Skilling, for example, pressured their accounting staff to use complex mark-to-market accounting on dubious assets and trade deals in order to meet shareholder expectations. Enron's deceit eventually led to the bankruptcy of the Enron company and later dissolution of their auditing company Arther Anderson in 2001. During the SEC investigation, Enron's email data became a key part of the prosecution's evidence of wrongdoing. This data,combined with the Enron employee financial data, allowed the SEC to bring the complicit executives to justice. The data then became public for anyone to explore.

In this project, I am going to use machine learning to build a 'person of interest' classifier based on the email and financial data from the Enron scandal. NOTE:Because of the relatively few positive matches (Enron employees charged), accuracy will not be the most important evaluation parameter. Also, due to the large amount of features, a feature-reduction algorithm will be used.

It is important to understand why machine learning is used in this case. Not all non-deterministic problems need to use machine learning to find a solution. For example, finding a few outliers or identifying an 'optimal' solution to a 2D problem can be done by a human relatively easily, and would be easier than coding a machine learning algorithm to predict the best solution(s). When we start exploring large datasets, or data where correlative power is low, and we want answers to 'complex' questions like what happens in a diverse ecosystem or who is the sender of these emails, machine learning can help. For this enron dataset, we have many high-dimensional low-correlative data points, and we would like to answer a simple but general binary classification question: who are the persons of interest? Machine learning is perfect for this type of data and question.

FILES:
EnronIdentification.html : Project summary & steps
poi_id.py : Final ML optimized algorithm
